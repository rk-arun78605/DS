import os
import hashlib
import json
import re
import string
import random
from datetime import datetime, timedelta, date

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings
from django.db.models import Q, Sum, Count, Min
from django.contrib import messages

from .models import ShopStaff, ItemMaster, Incident, MediaRegistry, ShopLocation


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def generate_incident_number(shop_code: str) -> str:
    """Generate unique incident number: WI-SPN-20260421-XXXX. Counts distinct incident_nos."""
    today = date.today().strftime('%Y%m%d')
    prefix = f"WI-{shop_code.upper()}-{today}-"
    # Count DISTINCT incident_nos today for this shop (not individual item rows)
    count = (
        Incident.objects.filter(shop_code=shop_code.upper(), submit_date__date=date.today())
        .values('incident_no').distinct().count()
    )
    serial = str(count + 1).zfill(4)
    candidate = f"{prefix}{serial}"
    while Incident.objects.filter(incident_no=candidate).exists():
        serial = str(int(serial) + 1).zfill(4)
        candidate = f"{prefix}{serial}"
    return candidate


def compute_file_hash(file_obj) -> str:
    """Layer 1 — SHA-256 exact byte match."""
    sha256 = hashlib.sha256()
    for chunk in file_obj.chunks():
        sha256.update(chunk)
    file_obj.seek(0)
    return sha256.hexdigest()


def extract_photo_fingerprints(file_obj):
    """
    Layer 2 — EXIF metadata signature.
    Layer 3 — Perceptual hash (pHash).

    Returns (exif_sig, phash_str) — either can be None if extraction fails.

    EXIF sig: SHA-256 of combined DateTimeOriginal + GPS + camera Make/Model + dimensions.
    pHash: 64-bit visual hash from imagehash — matches even if resized/cropped/brightness adjusted.
    Hamming distance <= 10 = visually similar.
    """
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    import imagehash

    exif_sig = None
    phash_str = None

    try:
        file_obj.seek(0)
        img = Image.open(file_obj)
        img.load()

        # ── Layer 3: perceptual hash ──────────────────────────────────
        phash_str = str(imagehash.phash(img))

        # ── Layer 2: EXIF fingerprint ─────────────────────────────────
        raw_exif = img._getexif() if hasattr(img, '_getexif') else None
        if raw_exif:
            tag_data = {TAGS.get(k, k): v for k, v in raw_exif.items()}
            parts = []
            # Date/time photo was actually taken
            for dt_field in ('DateTimeOriginal', 'DateTime', 'DateTimeDigitized'):
                if dt_field in tag_data:
                    parts.append(f"{dt_field}={tag_data[dt_field]}")
                    break
            # Camera identity
            for field in ('Make', 'Model', 'LensModel', 'Software'):
                if field in tag_data:
                    parts.append(f"{field}={tag_data[field]}")
            # GPS coordinates (rounded to 4 decimal places to tolerate minor drift)
            if 'GPSInfo' in tag_data:
                gps_raw = tag_data['GPSInfo']
                gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()} if isinstance(gps_raw, dict) else {}
                for g in ('GPSLatitude', 'GPSLongitude'):
                    if g in gps:
                        parts.append(f"{g}={gps[g]}")
            # Image dimensions
            parts.append(f"W={img.width},H={img.height}")
            if parts:
                sig_str = '|'.join(parts)
                exif_sig = hashlib.sha256(sig_str.encode()).hexdigest()
    except Exception:
        pass  # EXIF/pHash extraction is best-effort, never block the upload
    finally:
        file_obj.seek(0)

    return exif_sig, phash_str


def extract_video_fingerprint(file_obj) -> str | None:
    """
    Layer 2 for video — metadata signature from file header bytes + size.
    SHA-256 of: file_size + first-4KB + last-4KB.
    Catches re-encoded videos differently from exact hash but catches
    same-content videos with different containers or minor re-encodes.
    """
    try:
        file_obj.seek(0, 2)          # seek to end
        file_size = file_obj.tell()
        file_obj.seek(0)
        header = file_obj.read(4096)
        file_obj.seek(max(0, file_size - 4096))
        trailer = file_obj.read(4096)
        file_obj.seek(0)
        sig = hashlib.sha256(
            str(file_size).encode() + b'|' + header + b'|' + trailer
        ).hexdigest()
        return sig
    except Exception:
        file_obj.seek(0)
        return None


def save_media_file(file_obj, shop_code: str, incident_no: str, media_type: str) -> str:
    """
    Save uploaded file ONLY to network drive (no local copy on D: drive).
    Returns the UNC file path stored in DB.
    Raises OSError if network drive is unreachable.
    """
    net_root = getattr(settings, 'INCIDENT_MEDIA_ROOT', r'\\10.10.0.30\mis\wastagecontrol')
    ext = os.path.splitext(file_obj.name)[1].lower()
    filename = f"{incident_no}_{media_type}{ext}"
    net_dir = os.path.join(net_root, shop_code, incident_no)
    os.makedirs(net_dir, exist_ok=True)
    net_path = os.path.join(net_dir, filename)
    with open(net_path, 'wb+') as f:
        for chunk in file_obj.chunks():
            f.write(chunk)
    # Return relative path (from network root) for DB storage
    return os.path.join(shop_code, incident_no, filename)


def get_managed_shops(staff) -> list:
    """
    Return shop codes this user oversees.
    Management role always gets ALL shops — same as MGMT001.
    """
    if staff.role == 'management':
        all_shops = list(
            ShopStaff.objects.values_list('shop_code', flat=True).distinct()
        )
        return [s.upper() for s in all_shops if s]

    shops = [staff.shop_code.upper()]
    if getattr(staff, 'managed_shops', None):
        extra = [s.strip().upper() for s in staff.managed_shops.split(',') if s.strip()]
        shops.extend(extra)
    return list(set(shops))


def get_staff_profile(user):
    """Return ShopStaff profile for logged-in user or None."""
    try:
        return ShopStaff.objects.get(user=user)
    except ShopStaff.DoesNotExist:
        return None


# ─────────────────────────────────────────────────────────────
# PDF EVIDENCE VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_pdf_evidence(pdf_file, expected_pages: int = 0):
    """
    Validates a PDF evidence file:
      1. No duplicate pages within the PDF (pHash Hamming distance ≤ 10)
      2. No pages matching previously uploaded photos in MediaRegistry
    (Page count check removed — any number of pages is accepted.)

    Returns (is_valid: bool, error: str | None, page_data: list of (phash, page_sha256))
    """
    import hashlib as _hashlib
    from io import BytesIO
    from itertools import combinations

    try:
        import fitz
        import imagehash
        from PIL import Image
    except ImportError as e:
        return False, f"PDF validation requires PyMuPDF and Pillow: {e}", []

    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
    except Exception as e:
        return False, f"Cannot read PDF: {e}", []

    # (Page count validation removed — any number of pages accepted.)

    # Render each page to PNG → SHA-256 + pHash
    page_data = []
    for i in range(page_count):
        pix      = doc[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        page_sha = _hashlib.sha256(img_bytes).hexdigest()
        try:
            img  = Image.open(BytesIO(img_bytes))
            ph   = str(imagehash.phash(img))
        except Exception:
            ph = ''
        page_data.append((ph, page_sha))
    doc.close()

    # Within-PDF duplicate check
    valid_hashes = [(i, imagehash.hex_to_hash(ph)) for i, (ph, _) in enumerate(page_data) if ph]
    for (i, h1), (j, h2) in combinations(valid_hashes, 2):
        if h1 - h2 <= 10:
            return False, (
                f"Pages {i+1} and {j+1} appear to be the same image. "
                f"Each page must be a unique, original photo (one per item)."
            ), []

    # Check against existing MediaRegistry (external duplicate check)
    existing_phashes = list(
        MediaRegistry.objects.filter(media_type='photo')
        .exclude(perceptual_hash__isnull=True).exclude(perceptual_hash='')
        .values_list('perceptual_hash', 'incident_no', 'uploaded_at')
    )
    for page_num, (ph, page_sha) in enumerate(page_data, 1):
        # Layer 1: exact page SHA-256
        existing_exact = MediaRegistry.objects.filter(file_hash=page_sha).first()
        if existing_exact:
            return False, (
                f"Page {page_num} of the PDF is an exact duplicate of evidence already uploaded "
                f"(Incident: {existing_exact.incident_no}, "
                f"{existing_exact.uploaded_at.strftime('%d %b %Y')}). "
                f"Please use original, unsubmitted photos."
            ), []
        # Layer 3: perceptual hash
        if ph:
            try:
                new_ph = imagehash.hex_to_hash(ph)
                for stored_ph_str, orig_inc, orig_up in existing_phashes:
                    try:
                        if new_ph - imagehash.hex_to_hash(stored_ph_str) <= 10:
                            return False, (
                                f"Page {page_num} visually matches a previously uploaded photo "
                                f"(Incident: {orig_inc}, {orig_up.strftime('%d %b %Y')}). "
                                f"Please use original photos only."
                            ), []
                    except Exception:
                        continue
            except Exception:
                pass

    return True, None, page_data


def _check_item_photo_duplicate(ph_file):
    """
    Run 3-layer duplicate check on a per-item photo.
    Returns (is_duplicate: bool, error_message: str)
    """
    file_hash = compute_file_hash(ph_file)
    existing = MediaRegistry.objects.filter(file_hash=file_hash).first()
    if existing:
        return True, (
            f"Exact duplicate of evidence from incident {existing.incident_no} "
            f"({existing.uploaded_at.strftime('%d %b %Y')}). Use an original photo."
        )

    exif_sig, phash_str = extract_photo_fingerprints(ph_file)
    if phash_str:
        try:
            import imagehash
            new_ph = imagehash.hex_to_hash(phash_str)
            for stored_ph_str, orig_inc, orig_up in MediaRegistry.objects.filter(
                    media_type='photo').exclude(perceptual_hash__isnull=True
                    ).exclude(perceptual_hash='').values_list(
                    'perceptual_hash', 'incident_no', 'uploaded_at'):
                try:
                    if new_ph - imagehash.hex_to_hash(stored_ph_str) <= 10:
                        return True, (
                            f"Visually matches a previously uploaded photo "
                            f"(Incident: {orig_inc}, {orig_up.strftime('%d %b %Y')})."
                        )
                except Exception:
                    continue
        except Exception:
            pass
    return False, ''


# ─────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    error = None
    if request.method == 'POST':
        emp_id        = request.POST.get('emp_id', '').strip()
        password      = request.POST.get('password', '').strip()
        selected_role = request.POST.get('selected_role', 'staff').strip()

        # Find staff by emp_id
        try:
            staff = ShopStaff.objects.get(emp_id=emp_id, is_active=True)
        except ShopStaff.DoesNotExist:
            error = "Invalid Employee ID or password."
            return render(request, 'wastage/login.html', {'error': error})

        if not staff.user:
            error = "Account not fully configured. Contact admin."
            return render(request, 'wastage/login.html', {'error': error})

        # Validate selected role matches the staff's actual role
        # 'manager' kept for backward compat with existing DB records
        ROLE_MAP = {
            'staff':      ('staff',),
            'supervisor': ('supervisor', 'manager'),      # old 'manager' rows map here
            'management': ('management',),
        }
        is_mgmt_user = (staff.user.is_staff if staff.user else False) or (staff.role == 'management')

        if selected_role == 'management' and not is_mgmt_user:
            error = "You do not have Management access. Please select the correct role."
            return render(request, 'wastage/login.html', {'error': error})

        if selected_role != 'management':
            allowed_db_roles = ROLE_MAP.get(selected_role, ())
            if staff.role not in allowed_db_roles:
                actual_label = dict(ShopStaff.ROLE_CHOICES).get(staff.role, staff.role)
                error = f"Your role is '{actual_label}'. Please select the correct role."
                return render(request, 'wastage/login.html', {'error': error})

        user = authenticate(request, username=staff.user.username, password=password)
        if user is not None:
            login(request, user)
            request.session['selected_role'] = selected_role
            if _is_management(user) or selected_role == 'management':
                return redirect('mgmt_dashboard')
            return redirect('dashboard')
        else:
            error = "Invalid Employee ID or password."

    return render(request, 'wastage/login.html', {'error': error})


def logout_view(request):
    logout(request)
    return redirect('login')


def reset_password_view(request):
    """
    Password reset from the login page — no active session required.
    Verifies Employee ID + current password, then sets the new password.
    """
    if request.method != 'POST':
        return redirect('login')

    emp_id     = request.POST.get('emp_id', '').strip()
    old_pw     = request.POST.get('old_password', '').strip()
    new_pw     = request.POST.get('new_password', '').strip()
    confirm_pw = request.POST.get('confirm_password', '').strip()

    def fail(msg):
        """Re-render login page with modal open and error inside it."""
        return render(request, 'wastage/login.html', {'reset_error': msg})

    if not all([emp_id, old_pw, new_pw, confirm_pw]):
        return fail("All fields are required.")
    if new_pw != confirm_pw:
        return fail("New passwords do not match.")
    if len(new_pw) < 6:
        return fail("New password must be at least 6 characters.")

    try:
        staff = ShopStaff.objects.get(emp_id=emp_id, is_active=True)
    except ShopStaff.DoesNotExist:
        return fail("Employee ID not found or account is inactive.")

    if not staff.user:
        return fail("Account is not fully configured. Contact your Store Manager.")

    if not staff.user.check_password(old_pw):
        return fail("Current password is incorrect.")

    if old_pw == new_pw:
        return fail("New password must be different from the current password.")

    staff.user.set_password(new_pw)
    staff.user.save()

    messages.success(request, "Password reset successfully. Please sign in with your new password.")
    return redirect('login')


@login_required
def change_password_view(request):
    staff = get_staff_profile(request.user)
    success = False
    error = None

    if request.method == 'POST':
        old_pw = request.POST.get('old_password', '')
        new_pw = request.POST.get('new_password', '')
        confirm_pw = request.POST.get('confirm_password', '')

        if not request.user.check_password(old_pw):
            error = "Current password is incorrect."
        elif new_pw != confirm_pw:
            error = "New passwords do not match."
        elif len(new_pw) < 6:
            error = "Password must be at least 6 characters."
        else:
            request.user.set_password(new_pw)
            request.user.save()
            update_session_auth_hash(request, request.user)
            success = True

    return render(request, 'wastage/change_password.html', {
        'staff': staff,
        'success': success,
        'error': error,
    })


# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    staff = get_staff_profile(request.user)
    if not staff:
        return redirect('logout')

    today = date.today()

    if staff.is_approver:
        # Supervisor / Manager view — cover ALL managed shops
        shops = get_managed_shops(staff)

        pending_count = Incident.objects.filter(
            shop_code__in=shops,
            status='Pending'
        ).count()

        today_incidents = Incident.objects.filter(
            shop_code__in=shops,
            submit_date__date=today
        ).order_by('-submit_date')[:10]

        today_value = Incident.objects.filter(
            shop_code__in=shops,
            submit_date__date=today
        ).aggregate(total=Sum('total_value'))['total'] or 0

        # 48hr late submission check — only shown to supervisor
        yesterday = today - timedelta(days=1)
        missing_yesterday = not Incident.objects.filter(
            shop_code__in=shops,
            submit_date__date=yesterday
        ).exists()
        deadline = datetime.combine(yesterday + timedelta(days=2), datetime.min.time())
        hours_left = max(0, int((deadline - datetime.now()).total_seconds() / 3600))

        ctx = {
            'staff': staff,
            'today': today,
            'pending_count': pending_count,
            'today_incidents': today_incidents,
            'today_value': today_value,
            'missing_yesterday': missing_yesterday,
            'hours_left': hours_left,
            'is_approver': True,
        }
    else:
        # Staff view
        my_today = Incident.objects.filter(
            submitted_by=staff.emp_id,
            submit_date__date=today
        ).order_by('-submit_date')[:5]

        ctx = {
            'staff': staff,
            'today': today,
            'my_today': my_today,
            'is_approver': False,
        }

    return render(request, 'wastage/dashboard.html', ctx)


# ─────────────────────────────────────────────────────────────
# STAFF: SUBMIT INCIDENT
# ─────────────────────────────────────────────────────────────

@login_required
@login_required
def submit_check_photo(request):
    """
    AJAX endpoint: run 3-layer duplicate check on a candidate photo before form submission.
    No files saved, no DB writes — pure check.
    Returns JSON: {duplicate: false} or {duplicate: true, layer:..., original_incident:..., uploaded_at:...}
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    file_obj = request.FILES.get('photo')
    if not file_obj:
        return JsonResponse({'error': 'No file provided'}, status=400)

    is_pdf = file_obj.name.lower().endswith('.pdf')

    # ── Layer 1: exact SHA-256 ─────────────────────────────────────────────────
    file_hash = compute_file_hash(file_obj)
    existing = MediaRegistry.objects.filter(file_hash=file_hash).first()
    if existing:
        return JsonResponse({
            'duplicate': True,
            'layer': 'Exact file match (SHA-256)',
            'original_incident': existing.incident_no,
            'uploaded_at': existing.uploaded_at.strftime('%d %b %Y %H:%M'),
        })

    # ── Layers 2 & 3: image-only (skip for PDFs) ──────────────────────────────
    if is_pdf:
        return JsonResponse({'duplicate': False})

    exif_sig, phash_str = extract_photo_fingerprints(file_obj)

    if exif_sig:
        existing_exif = MediaRegistry.objects.filter(
            exif_signature=exif_sig, media_type='photo'
        ).first()
        if existing_exif:
            return JsonResponse({
                'duplicate': True,
                'layer': 'Same photo — EXIF metadata match (same device, date & location)',
                'original_incident': existing_exif.incident_no,
                'uploaded_at': existing_exif.uploaded_at.strftime('%d %b %Y %H:%M'),
            })

    # ── Layer 3: perceptual hash (visual similarity) ───────────────────────────
    if phash_str:
        all_photo_phashes = MediaRegistry.objects.filter(
            media_type='photo'
        ).exclude(perceptual_hash__isnull=True).exclude(perceptual_hash='').values_list(
            'perceptual_hash', 'incident_no', 'uploaded_at'
        )
        import imagehash
        try:
            new_phash = imagehash.hex_to_hash(phash_str)
            for stored_phash_str, orig_incident, orig_uploaded in all_photo_phashes:
                try:
                    stored_phash = imagehash.hex_to_hash(stored_phash_str)
                    distance = new_phash - stored_phash
                    if distance <= 10:
                        return JsonResponse({
                            'duplicate': True,
                            'layer': f'Visually similar photo (perceptual hash distance: {distance}/64)',
                            'original_incident': orig_incident,
                            'uploaded_at': orig_uploaded.strftime('%d %b %Y %H:%M'),
                        })
                except Exception:
                    continue
        except Exception:
            pass

    return JsonResponse({'duplicate': False})


def submit_incident(request):
    staff = get_staff_profile(request.user)
    if not staff:
        return redirect('logout')

    if request.method == 'POST':
        action = request.POST.get('action', 'submit')  # Change 8: 'draft' or 'submit'
        department = request.POST.get('department', '').strip()
        remarks = request.POST.get('remarks', '').strip()
        item_count = min(int(request.POST.get('item_count', '1') or '1'), 50)

        errors = []

        if not department:
            errors.append("Please select a department.")

        # Parse all item rows (Change 1: per-item reason)
        items_data = []
        seen_item_reason = {}  # Change 5: duplicate item+reason check
        for i in range(item_count):
            item_name = request.POST.get(f'item_name_{i}', '').strip()
            item_code = request.POST.get(f'item_code_{i}', 'AAAAA').strip() or 'AAAAA'
            category = request.POST.get(f'category_{i}', '').strip()
            uom = request.POST.get(f'uom_{i}', '').strip()
            quantity_raw = request.POST.get(f'quantity_{i}', '0')
            selling_price_raw = request.POST.get(f'selling_price_{i}', '0')
            row_reason = request.POST.get(f'reason_{i}', '').strip()  # Change 1: per-item reason
            row_remarks = request.POST.get(f'remarks_{i}', '').strip()

            if not item_name:
                errors.append(f"Row {i + 1}: Item name is required.")
                continue
            if not category:
                errors.append(f"Row {i + 1}: Category could not be determined. Please re-select the item.")
                continue
            if not row_reason and action != 'draft':
                errors.append(f"Row {i + 1}: Please select a reason.")
                continue
            if row_reason == 'Others' and not row_remarks and action != 'draft':
                errors.append(f"Row {i + 1}: Remarks are mandatory when reason is 'Others'.")
                continue
            try:
                quantity = float(quantity_raw)
                if quantity <= 0 and action != 'draft':
                    errors.append(f"Row {i + 1}: Quantity must be greater than zero.")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Row {i + 1}: Invalid quantity.")
                continue
            try:
                selling_price = float(selling_price_raw)
            except (ValueError, TypeError):
                selling_price = 0.0

            # Change 5: duplicate item+reason check within same submission
            dup_key = (item_code, row_reason)
            if dup_key in seen_item_reason and action != 'draft':
                errors.append(
                    f"Item '{item_name}' with reason '{row_reason}' is already added to this incident."
                )
                continue
            seen_item_reason[dup_key] = i

            # Change 4: expiry date/photo
            expiry_date_raw = request.POST.get(f'expiry_date_{i}', '').strip()
            expiry_date = None
            if row_reason == 'Expired' and expiry_date_raw:
                try:
                    from datetime import date as date_cls
                    expiry_date = date_cls.fromisoformat(expiry_date_raw)
                except ValueError:
                    pass

            items_data.append({
                'item_code': item_code,
                'item_name': item_name,
                'category': category,
                'quantity': quantity,
                'uom': uom,
                'selling_price': selling_price,
                'reason': row_reason,
                'remarks': row_remarks if row_remarks else (remarks if remarks else None),
                'expiry_date': expiry_date,
                'expiry_photo_key': f'expiry_photo_{i}',  # file input name
                'row_idx': i,
            })

        if not items_data and not errors:
            errors.append("Please add at least one item.")

        # ── Evidence photo validation ──────────────────────────────────────────
        # Per-item photos are optional — no validation on those.
        # Main photo (Section 3) is required for real submissions.
        item_photos = [request.FILES.get(f'item_photo_{item["row_idx"]}') for item in items_data]
        has_photo   = [ph is not None for ph in item_photos]
        pdf_file    = request.FILES.get('incident_pdf')
        pdf_phashes = []
        main_photo  = request.FILES.get('photo')

        if action != 'draft' and not errors:
            if not main_photo:
                errors.append("Photo evidence is required. Please attach a photo before submitting.")
            else:
                # Duplicate check on the main photo
                is_dup, dup_msg = _check_item_photo_duplicate(main_photo)
                if is_dup:
                    errors.append(f"Photo evidence: {dup_msg}")

        if errors:
            return render(request, 'wastage/submit_incident.html', {
                'staff': staff,
                'today': date.today(),
                'departments': Incident.DEPARTMENT_CHOICES,
                'reasons': Incident.REASON_CHOICES,
                'errors': errors,
                'post': request.POST,
                'item_count_range': range(item_count),
            })

        # Determine status — Draft or Pending
        incident_status = 'Draft' if action == 'draft' else 'Pending'

        # Only 1 draft allowed per staff
        if incident_status == 'Draft':
            if Incident.objects.filter(submitted_by=staff.emp_id, status='Draft').exists():
                return render(request, 'wastage/submit_incident.html', {
                    'staff': staff,
                    'today': date.today(),
                    'departments': Incident.DEPARTMENT_CHOICES,
                    'reasons': Incident.REASON_CHOICES,
                    'errors': ["You already have a saved draft. Please submit or delete it before saving a new one."],
                    'post': request.POST,
                    'item_count_range': range(item_count),
                })

        # Draft gets no incident_no; real submissions generate one
        incident_no = '' if incident_status == 'Draft' else generate_incident_number(staff.shop_code)
        created_pks = []
        for item in items_data:
            inc = Incident.objects.create(
                incident_no=incident_no,
                shop_code=staff.shop_code,
                shop_name=staff.shop_name,
                department=department,
                submitted_by=staff.emp_id,
                submitted_name=staff.emp_name,
                item_code=item['item_code'],
                item_name=item['item_name'],
                category=item['category'],
                quantity=item['quantity'],
                uom=item['uom'],
                selling_price=item['selling_price'],
                reason=item['reason'],
                remarks=item['remarks'],
                status=incident_status,
                expiry_date=item['expiry_date'],
            )
            created_pks.append(inc.pk)

            # Change 4: handle expiry photo upload
            expiry_photo = request.FILES.get(item['expiry_photo_key'])
            if expiry_photo and item['reason'] == 'Expired':
                try:
                    file_hash = compute_file_hash(expiry_photo)
                    net_root = getattr(settings, 'INCIDENT_MEDIA_ROOT', r'\\10.10.0.30\mis\wastagecontrol')
                    ext = os.path.splitext(expiry_photo.name)[1].lower()
                    row_idx = item['row_idx']
                    filename = f"expiry_{row_idx}{ext}"
                    net_dir = os.path.join(net_root, staff.shop_code, incident_no)
                    os.makedirs(net_dir, exist_ok=True)
                    net_path = os.path.join(net_dir, filename)
                    with open(net_path, 'wb+') as f:
                        for chunk in expiry_photo.chunks():
                            f.write(chunk)
                    rel_path = os.path.join(staff.shop_code, incident_no, filename)
                    Incident.objects.filter(pk=inc.pk).update(
                        expiry_photo_path=rel_path,
                        expiry_photo_hash=file_hash,
                    )
                except Exception:
                    pass  # non-blocking

        # ── Save per-item photos ────────────────────────────────────────────────
        net_root = getattr(settings, 'INCIDENT_MEDIA_ROOT', r'\\10.10.0.30\mis\wastagecontrol')

        if all(has_photo) and action != 'draft':
            for item, inc_pk, ph_file in zip(items_data, created_pks, item_photos):
                if not ph_file:
                    continue
                try:
                    file_hash        = compute_file_hash(ph_file)
                    exif_sig, phash_str = extract_photo_fingerprints(ph_file)
                    ext      = os.path.splitext(ph_file.name)[1].lower() or '.jpg'
                    filename = f"item_photo_{item['row_idx']}{ext}"
                    net_dir  = os.path.join(net_root, staff.shop_code, incident_no)
                    os.makedirs(net_dir, exist_ok=True)
                    with open(os.path.join(net_dir, filename), 'wb+') as f:
                        for chunk in ph_file.chunks():
                            f.write(chunk)
                    rel_path = os.path.join(staff.shop_code, incident_no, filename)
                    Incident.objects.filter(pk=inc_pk).update(
                        item_photo_path=rel_path,
                        item_photo_hash=file_hash,
                        item_photo_phash=phash_str or '',
                    )
                    MediaRegistry.objects.create(
                        incident_no=incident_no,
                        media_type='photo',
                        file_hash=file_hash,
                        exif_signature=exif_sig,
                        perceptual_hash=phash_str,
                        file_path=rel_path,
                    )
                except Exception:
                    pass

        # ── Save PDF evidence ──────────────────────────────────────────────────
        elif pdf_file and pdf_phashes and action != 'draft':
            try:
                import hashlib as _hl
                pdf_file.seek(0)
                pdf_bytes   = pdf_file.read()
                pdf_sha256  = _hl.sha256(pdf_bytes).hexdigest()
                filename    = f"evidence_{incident_no}.pdf"
                net_dir     = os.path.join(net_root, staff.shop_code, incident_no)
                os.makedirs(net_dir, exist_ok=True)
                with open(os.path.join(net_dir, filename), 'wb') as f:
                    f.write(pdf_bytes)
                rel_path = os.path.join(staff.shop_code, incident_no, filename)
                # Store on all incident rows (same incident_no)
                Incident.objects.filter(incident_no=incident_no).update(
                    photo_path=rel_path,
                    photo_hash=pdf_sha256,
                )
                # Register each page's hash for future duplicate detection
                for ph, page_sha in pdf_phashes:
                    MediaRegistry.objects.create(
                        incident_no=incident_no,
                        media_type='photo',
                        file_hash=page_sha,
                        perceptual_hash=ph,
                        file_path=rel_path,
                    )
            except Exception:
                pass

        # ── Save main photo ────────────────────────────────────────────────────
        photo = request.FILES.get('photo')
        if photo:
            try:
                file_hash = compute_file_hash(photo)
                exif_sig, phash_str = extract_photo_fingerprints(photo)
                file_path = save_media_file(photo, staff.shop_code, incident_no, 'photo')
                MediaRegistry.objects.create(
                    incident_no=incident_no,
                    media_type='photo',
                    file_hash=file_hash,
                    exif_signature=exif_sig,
                    perceptual_hash=phash_str,
                    file_path=file_path,
                )
                Incident.objects.filter(incident_no=incident_no).update(
                    photo_path=file_path,
                    photo_hash=file_hash,
                    photo_exif_sig=exif_sig,
                    photo_phash=phash_str,
                )
            except Exception:
                pass

        # ── Save short video (optional) ────────────────────────────────────────
        video = request.FILES.get('video')
        if video and action != 'draft':
            MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50 MB
            if video.size <= MAX_VIDEO_BYTES:
                try:
                    video_hash = compute_file_hash(video)
                    video_path = save_media_file(video, staff.shop_code, incident_no, 'video')
                    MediaRegistry.objects.create(
                        incident_no=incident_no,
                        media_type='video',
                        file_hash=video_hash,
                        file_path=video_path,
                    )
                    Incident.objects.filter(incident_no=incident_no).update(
                        video_path=video_path,
                        video_hash=video_hash,
                    )
                except Exception:
                    pass

        if action == 'draft':
            messages.success(
                request,
                f"Draft saved ({len(items_data)} item{'s' if len(items_data) > 1 else ''}). "
                f"An incident number will be assigned when you submit."
            )
            return redirect('submit_incident')  # Stay on submit page so they see the draft banner
        else:
            messages.success(
                request,
                f"Wastage incident submitted! Incident No: <strong>{incident_no}</strong> "
                f"({len(items_data)} item{'s' if len(items_data) > 1 else ''})"
            )
        return redirect('my_incidents')

    # Show existing draft banner if one exists
    existing_draft = Incident.objects.filter(
        submitted_by=staff.emp_id, status='Draft'
    ).order_by('pk').first()

    return render(request, 'wastage/submit_incident.html', {
        'staff': staff,
        'today': date.today(),
        'departments': Incident.DEPARTMENT_CHOICES,
        'reasons': Incident.REASON_CHOICES,
        'errors': [],
        'post': {},
        'item_count_range': range(1),
        'existing_draft': existing_draft,
    })


@login_required
def my_incidents(request):
    staff = get_staff_profile(request.user)
    if not staff:
        return redirect('logout')

    date_filter = request.GET.get('date', '')
    # Exclude drafts — they are shown as a banner on the Submit page, not here
    qs = Incident.objects.filter(submitted_by=staff.emp_id).exclude(status='Draft')
    if date_filter:
        try:
            fd = datetime.strptime(date_filter, '%Y-%m-%d').date()
            qs = qs.filter(submit_date__date=fd)
        except ValueError:
            pass

    # Deduplicate: one entry per incident_no with aggregated totals
    groups = (
        qs.values('incident_no')
        .annotate(min_pk=Min('pk'), total_sum=Sum('total_value'), item_count=Count('pk'))
        .order_by('-min_pk')
    )
    rep_map = {inc.pk: inc for inc in Incident.objects.filter(pk__in=[g['min_pk'] for g in groups])}
    incidents = []
    for g in groups:
        inc = rep_map[g['min_pk']]
        inc.total_value_sum = g['total_sum']
        inc.item_count = g['item_count']
        incidents.append(inc)

    return render(request, 'wastage/my_incidents.html', {
        'staff': staff,
        'incidents': incidents,
        'today': date.today(),
        'date_filter': date_filter,
    })


# ─────────────────────────────────────────────────────────────
# SUPERVISOR / MANAGER: APPROVALS
# ─────────────────────────────────────────────────────────────

@login_required
def approvals_list(request):
    staff = get_staff_profile(request.user)
    if not staff or not staff.is_approver:
        messages.error(request, "You do not have access to this page.")
        return redirect('dashboard')

    # Show incidents for ALL managed shops
    shops = get_managed_shops(staff)
    filter_date = request.GET.get('date', str(date.today()))
    filter_status = request.GET.get('status', 'Pending')

    # Exclude Drafts from supervisor view — drafts are staff-only until submitted
    incidents = Incident.objects.filter(shop_code__in=shops).exclude(status='Draft')

    if filter_date:
        try:
            fd = datetime.strptime(filter_date, '%Y-%m-%d').date()
            incidents = incidents.filter(submit_date__date=fd)
        except ValueError:
            pass

    if filter_status and filter_status != 'All':
        incidents = incidents.filter(status=filter_status)

    incidents = incidents.order_by('-submit_date')

    # Deduplicate by incident_no — show one row per incident with aggregated totals
    groups = (
        incidents
        .values('incident_no')
        .annotate(min_pk=Min('pk'), total_sum=Sum('total_value'), item_count=Count('pk'))
        .order_by('-min_pk')
    )
    rep_map = {inc.pk: inc for inc in Incident.objects.filter(pk__in=[g['min_pk'] for g in groups])}
    incidents_list = []
    for g in groups:
        inc = rep_map[g['min_pk']]
        inc.total_value_sum = g['total_sum']
        inc.item_count = g['item_count']
        incidents_list.append(inc)

    today_total = (
        Incident.objects.filter(shop_code__in=shops, submit_date__date=date.today())
        .aggregate(total=Sum('total_value'))['total'] or 0
    )

    return render(request, 'wastage/approvals_list.html', {
        'staff': staff,
        'incidents': incidents_list,
        'filter_date': filter_date,
        'filter_status': filter_status,
        'today_total': today_total,
        'today': date.today(),
        'status_choices': [('All', 'All'), ('Pending', 'Pending'), ('Approved', 'Approved'), ('Rejected', 'Rejected')],
    })


@login_required
def approval_detail(request, pk):
    staff = get_staff_profile(request.user)
    if not staff or not staff.is_approver:
        return redirect('dashboard')

    incident = get_object_or_404(Incident, pk=pk, shop_code__in=get_managed_shops(staff))
    # Load ALL item rows for this incident (multi-item support)
    items = list(Incident.objects.filter(incident_no=incident.incident_no).order_by('pk'))
    items_total = sum(i.total_value for i in items)

    return render(request, 'wastage/approval_detail.html', {
        'staff': staff,
        'incident': incident,
        'items': items,
        'items_total': items_total,
        'today': date.today(),
    })


@login_required
@require_POST
def upload_media(request, pk):
    staff = get_staff_profile(request.user)
    if not staff or not staff.is_approver:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    incident = get_object_or_404(Incident, pk=pk, shop_code__in=get_managed_shops(staff))

    if incident.status == 'Approved':
        return JsonResponse({'error': 'Incident already approved.'}, status=400)

    photo = request.FILES.get('photo')
    video = request.FILES.get('video')

    if not photo and not video:
        return JsonResponse({'error': 'Please upload at least one photo or video.'}, status=400)

    response_data = {}
    duplicate_info = []

    def _check_and_save(file_obj, media_type):
        """
        Run 3-layer duplicate detection then save.
        Returns (saved:bool, duplicate_detail:dict|None, error:str|None)
        """
        # ── Layer 1: exact SHA-256 ────────────────────────────────────
        file_hash = compute_file_hash(file_obj)
        existing = MediaRegistry.objects.filter(
            file_hash=file_hash
        ).exclude(incident_no=incident.incident_no).first()
        if existing:
            return False, {
                'type': media_type,
                'layer': 'Exact file match (SHA-256)',
                'original_incident': existing.incident_no,
                'uploaded_at': existing.uploaded_at.strftime('%d %b %Y %H:%M'),
            }, None

        exif_sig = None
        phash_str = None
        meta_sig = None

        if media_type == 'photo':
            # ── Layer 2: EXIF metadata fingerprint ───────────────────
            exif_sig, phash_str = extract_photo_fingerprints(file_obj)

            if exif_sig:
                existing_exif = MediaRegistry.objects.filter(
                    exif_signature=exif_sig, media_type='photo'
                ).exclude(incident_no=incident.incident_no).first()
                if existing_exif:
                    return False, {
                        'type': media_type,
                        'layer': 'Same photo — EXIF metadata match (same device, date & location)',
                        'original_incident': existing_exif.incident_no,
                        'uploaded_at': existing_exif.uploaded_at.strftime('%d %b %Y %H:%M'),
                    }, None

            # ── Layer 3: perceptual hash (visual similarity) ──────────
            if phash_str:
                # Compare pHash Hamming distance — ≤10 = visually similar
                all_photo_phashes = MediaRegistry.objects.filter(
                    media_type='photo'
                ).exclude(incident_no=incident.incident_no).exclude(
                    perceptual_hash__isnull=True
                ).exclude(perceptual_hash='').values_list(
                    'perceptual_hash', 'incident_no', 'uploaded_at'
                )
                import imagehash
                try:
                    new_phash = imagehash.hex_to_hash(phash_str)
                    for stored_phash_str, orig_incident, orig_uploaded in all_photo_phashes:
                        try:
                            stored_phash = imagehash.hex_to_hash(stored_phash_str)
                            distance = new_phash - stored_phash
                            if distance <= 10:  # ≤10/64 bits different = visually similar
                                return False, {
                                    'type': media_type,
                                    'layer': f'Visually similar photo (perceptual hash distance: {distance}/64)',
                                    'original_incident': orig_incident,
                                    'uploaded_at': orig_uploaded.strftime('%d %b %Y %H:%M'),
                                }, None
                        except Exception:
                            continue
                except Exception:
                    pass

        else:  # video
            # ── Layer 2 (video): metadata/structure fingerprint ───────
            meta_sig = extract_video_fingerprint(file_obj)
            if meta_sig:
                existing_meta = MediaRegistry.objects.filter(
                    exif_signature=meta_sig, media_type='video'
                ).exclude(incident_no=incident.incident_no).first()
                if existing_meta:
                    return False, {
                        'type': media_type,
                        'layer': 'Same video — structure & size fingerprint match',
                        'original_incident': existing_meta.incident_no,
                        'uploaded_at': existing_meta.uploaded_at.strftime('%d %b %Y %H:%M'),
                    }, None

        # ── All checks passed — delete old registry entry then save ──
        try:
            # Remove previous upload for this incident+type (supervisor changed file)
            MediaRegistry.objects.filter(
                incident_no=incident.incident_no, media_type=media_type
            ).delete()

            file_path = save_media_file(file_obj, incident.shop_code, incident.incident_no, media_type)
            MediaRegistry.objects.create(
                incident_no=incident.incident_no,
                media_type=media_type,
                file_hash=file_hash,
                exif_signature=exif_sig or meta_sig,
                perceptual_hash=phash_str,
                file_path=file_path,
            )
            # Update incident record
            if media_type == 'photo':
                incident.photo_path = file_path
                incident.photo_hash = file_hash
                incident.photo_exif_sig = exif_sig
                incident.photo_phash = phash_str
            else:
                incident.video_path = file_path
                incident.video_hash = file_hash
                incident.video_meta_sig = meta_sig
            return True, None, None
        except Exception as e:
            return False, None, str(e)

    # Process photo
    if photo:
        saved, dup, err = _check_and_save(photo, 'photo')
        if dup:
            duplicate_info.append(dup)
        elif err:
            response_data['photo_error'] = err
        else:
            response_data['photo_saved'] = True

    # Process video
    if video:
        saved, dup, err = _check_and_save(video, 'video')
        if dup:
            duplicate_info.append(dup)
        elif err:
            response_data['video_error'] = err
        else:
            response_data['video_saved'] = True

    if duplicate_info:
        return JsonResponse({
            'duplicates': duplicate_info,
            'message': 'Duplicate media detected. Please re-upload with original evidence.'
        }, status=409)

    incident.save(update_fields=[
        'photo_path', 'photo_hash', 'photo_exif_sig', 'photo_phash',
        'video_path', 'video_hash', 'video_meta_sig', 'updated_at',
    ])
    # Sync media fields to all other rows of the same incident (multi-item)
    sync_kwargs = {}
    if incident.photo_path:
        sync_kwargs.update({
            'photo_path': incident.photo_path, 'photo_hash': incident.photo_hash,
            'photo_exif_sig': incident.photo_exif_sig, 'photo_phash': incident.photo_phash,
        })
    if incident.video_path:
        sync_kwargs.update({
            'video_path': incident.video_path, 'video_hash': incident.video_hash,
            'video_meta_sig': incident.video_meta_sig,
        })
    if sync_kwargs:
        Incident.objects.filter(
            incident_no=incident.incident_no
        ).exclude(pk=incident.pk).update(**sync_kwargs)

    response_data['has_media'] = bool(incident.photo_path or incident.video_path)
    response_data['success'] = True
    return JsonResponse(response_data)


@login_required
@require_POST
def approve_incident(request, pk):
    """
    Change 7: Per-item approve/reject with individual remarks.
    Change 3: Save approved_quantity per item.
    POST form fields: item_action_{pk}, item_remark_{pk}, approved_quantity_{pk}
    """
    staff = get_staff_profile(request.user)
    if not staff or not staff.is_approver:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    incident = get_object_or_404(Incident, pk=pk, shop_code__in=get_managed_shops(staff))

    # Load all item rows for this incident
    all_items = list(Incident.objects.filter(incident_no=incident.incident_no))

    now = timezone.now()
    for item in all_items:
        action = request.POST.get(f'item_action_{item.pk}', 'Keep Pending').strip()
        remark = request.POST.get(f'item_remark_{item.pk}', '').strip()
        approved_qty_raw = request.POST.get(f'approved_quantity_{item.pk}', '').strip()

        approved_qty = None
        if approved_qty_raw:
            try:
                approved_qty = float(approved_qty_raw)
            except (ValueError, TypeError):
                approved_qty = None

        if action == 'Approve':
            item.status = 'Approved'
            item.approved_by = staff.emp_id
            item.approved_name = staff.emp_name
            item.approved_date = now
            item.supervisor_remark = remark
            if approved_qty is not None:
                item.approved_quantity = approved_qty
            item.save(update_fields=[
                'status', 'approved_by', 'approved_name', 'approved_date',
                'supervisor_remark', 'approved_quantity', 'updated_at',
            ])
        elif action == 'Reject':
            item.status = 'Rejected'
            item.approved_by = staff.emp_id
            item.approved_name = staff.emp_name
            item.approved_date = now
            item.supervisor_remark = remark
            item.save(update_fields=[
                'status', 'approved_by', 'approved_name', 'approved_date',
                'supervisor_remark', 'updated_at',
            ])
        # else 'Keep Pending' — no change

    return JsonResponse({'success': True, 'incident_no': incident.incident_no})


@login_required
@require_POST
def reject_incident(request, pk):
    staff = get_staff_profile(request.user)
    if not staff or not staff.is_approver:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    incident = get_object_or_404(Incident, pk=pk, shop_code__in=get_managed_shops(staff))

    if incident.status != 'Pending':
        return JsonResponse({'error': f'Incident is already {incident.status}.'}, status=400)

    try:
        body = json.loads(request.body)
        reason = body.get('reason', '').strip()
    except Exception:
        reason = ''

    if not reason:
        return JsonResponse({'error': 'Please provide a reason for rejection.'}, status=400)

    now = timezone.now()
    reject_note = f'\n[Rejected by {staff.emp_name}: {reason}]'
    # Bulk-update ALL item rows for this incident
    Incident.objects.filter(incident_no=incident.incident_no).update(
        status='Rejected',
        approved_by=staff.emp_id,
        approved_name=staff.emp_name,
        approved_date=now,
        updated_at=now,
    )
    for row in Incident.objects.filter(incident_no=incident.incident_no):
        row.remarks = ((row.remarks or '') + reject_note).strip()
        row.save(update_fields=['remarks'])

    return JsonResponse({'success': True, 'incident_no': incident.incident_no})


# ─────────────────────────────────────────────────────────────
# Change 8: EDIT DRAFT INCIDENT
# ─────────────────────────────────────────────────────────────

@login_required
def edit_incident(request, draft_pk):
    """Load a draft incident (by first-item pk) for editing and re-submission."""
    staff = get_staff_profile(request.user)
    if not staff:
        return redirect('logout')

    # Fetch the first draft item to get the shared identifier
    try:
        anchor = Incident.objects.get(pk=draft_pk, submitted_by=staff.emp_id, status='Draft')
    except Incident.DoesNotExist:
        messages.error(request, "Draft not found or already submitted.")
        return redirect('submit_incident')

    # All draft rows for this staff (same status=Draft, same department group)
    # Drafts have empty incident_no — fetch all Draft rows by this staff created in the same batch
    # We identify a "draft batch" by created_at proximity; simplest: all drafts by this staff
    items = list(
        Incident.objects.filter(
            submitted_by=staff.emp_id,
            status='Draft',
        ).order_by('pk')
    )

    if request.method == 'POST':
        # Delete all draft rows then re-submit
        Incident.objects.filter(submitted_by=staff.emp_id, status='Draft').delete()
        return submit_incident(request)

    return render(request, 'wastage/submit_incident.html', {
        'staff': staff,
        'today': date.today(),
        'departments': Incident.DEPARTMENT_CHOICES,
        'reasons': Incident.REASON_CHOICES,
        'errors': [],
        'post': {},
        'item_count_range': range(len(items)),
        'draft_items': items,
        'edit_mode': True,
    })


# ─────────────────────────────────────────────────────────────
# Change 9: WIP MASTER LIST
# ─────────────────────────────────────────────────────────────

@login_required
def wip_master_list(request):
    """List all WIP and Semifinished items — accessible to supervisors and managers."""
    staff = get_staff_profile(request.user)
    if not staff or not staff.is_approver:
        messages.error(request, "You do not have access to this page.")
        return redirect('dashboard')

    q = request.GET.get('q', '').strip()
    items = ItemMaster.objects.filter(
        category__in=['WIP', 'Semifinished'],
        is_active=True,
    )
    if q:
        items = items.filter(
            Q(item_name__icontains=q) | Q(item_code__icontains=q)
        )
    items = items.order_by('category', 'item_name')

    return render(request, 'wastage/wip_master_list.html', {
        'staff': staff,
        'items': items,
        'today': date.today(),
        'q': q,
    })


# ─────────────────────────────────────────────────────────────
# Change 10: GEOFENCING — CHECK LOCATION
# ─────────────────────────────────────────────────────────────

import math


def _haversine_meters(lat1, lon1, lat2, lon2):
    """Return distance in metres between two (lat, lon) points."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@require_POST
@login_required
def check_location(request):
    """
    Change 10: AJAX endpoint.
    POST: lat, lon (from navigator.geolocation)
    Returns JSON: {allowed: bool, distance_meters: float, message: str}
    Fail-open: if no ShopLocation record exists, allow submission.
    """
    try:
        lat = float(request.POST.get('lat', ''))
        lon = float(request.POST.get('lon', ''))
    except (ValueError, TypeError):
        return JsonResponse({'allowed': True, 'distance_meters': 0, 'message': 'Invalid coordinates — allowed by default.'})

    staff = get_staff_profile(request.user)
    if not staff:
        return JsonResponse({'allowed': True, 'distance_meters': 0, 'message': 'No staff profile.'})

    try:
        loc = ShopLocation.objects.get(shop_code=staff.shop_code.upper(), is_active=True)
    except ShopLocation.DoesNotExist:
        # Fail-open: no location record configured for this shop
        return JsonResponse({
            'allowed': True,
            'distance_meters': 0,
            'message': 'No location configured for your shop — submission allowed.',
        })

    distance = _haversine_meters(lat, lon, loc.latitude, loc.longitude)
    allowed = distance <= loc.radius_meters
    msg = (
        f"You are {distance:.0f}m from your shop location (allowed radius: {loc.radius_meters}m)."
        if not allowed
        else f"Location verified — {distance:.0f}m from shop."
    )
    return JsonResponse({'allowed': allowed, 'distance_meters': round(distance, 1), 'message': msg})


# ─────────────────────────────────────────────────────────────
# SERVE MEDIA (view uploaded photos/videos for approved incidents)
# ─────────────────────────────────────────────────────────────

@login_required
def serve_media(request, pk, media_type):
    """
    Serve a photo or video from the network drive so supervisors can view
    evidence on approved/rejected incidents without needing direct file access.
    """
    # MGMT users bypass shop restriction
    if _is_management(request.user):
        incident = get_object_or_404(Incident, pk=pk)
    else:
        staff = get_staff_profile(request.user)
        if not staff or not staff.is_approver:
            return HttpResponse(status=403)
        incident = get_object_or_404(Incident, pk=pk, shop_code__in=get_managed_shops(staff))

    ALLOWED = {
        'photo': {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                  '.png': 'image/png', '.webp': 'image/webp',
                  '.pdf': 'application/pdf'},
        'video': {'.mp4': 'video/mp4', '.mov': 'video/quicktime',
                  '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska',
                  '.pdf': 'application/pdf'},
    }
    if media_type not in ALLOWED:
        return HttpResponse(status=400)

    rel_path = incident.photo_path if media_type == 'photo' else incident.video_path
    if not rel_path:
        return HttpResponse(status=404)

    net_root = getattr(settings, 'INCIDENT_MEDIA_ROOT', r'\\10.10.0.30\mis\wastagecontrol')
    full_path = os.path.join(net_root, rel_path)

    if not os.path.exists(full_path):
        return HttpResponse(b'File not found on network drive.', status=404,
                            content_type='text/plain')

    ext = os.path.splitext(full_path)[1].lower()
    content_type = ALLOWED[media_type].get(ext)
    if not content_type:
        return HttpResponse(status=415)

    with open(full_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type=content_type)
    # Tell browser to display inline (not download)
    response['Content-Disposition'] = f'inline; filename="{os.path.basename(full_path)}"'
    return response


@login_required
def serve_item_expiry_photo(request, item_pk):
    """Serve a per-item expiry photo from the network drive for supervisor/management preview."""
    staff = get_staff_profile(request.user)
    if not staff or (not staff.is_approver and not _is_management(request.user)):
        return HttpResponse(status=403)

    if _is_management(request.user):
        item = get_object_or_404(Incident, pk=item_pk)
    else:
        item = get_object_or_404(Incident, pk=item_pk, shop_code__in=get_managed_shops(staff))

    if not item.expiry_photo_path:
        return HttpResponse(status=404)

    net_root = getattr(settings, 'INCIDENT_MEDIA_ROOT', r'\\10.10.0.30\mis\wastagecontrol')
    full_path = os.path.join(net_root, item.expiry_photo_path)

    if not os.path.exists(full_path):
        return HttpResponse(b'Expiry photo not found on network drive.', status=404,
                            content_type='text/plain')

    ALLOWED = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
               '.webp': 'image/webp', '.pdf': 'application/pdf'}
    ext = os.path.splitext(full_path)[1].lower()
    content_type = ALLOWED.get(ext)
    if not content_type:
        return HttpResponse(status=415)

    with open(full_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type=content_type)
    response['Content-Disposition'] = f'inline; filename="{os.path.basename(full_path)}"'
    return response


# ─────────────────────────────────────────────────────────────
# ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────

@login_required
def analytics_dashboard(request):
    from django.db.models.functions import TruncDate, ExtractWeekDay

    staff = get_staff_profile(request.user)
    if not staff or not _is_management(request.user):
        messages.error(request, "Analytics access is restricted to Management users only.")
        return redirect('dashboard')

    shops = get_managed_shops(staff)

    # ── Filters ────────────────────────────────────────────────
    dept_filter = request.GET.get('dept', '').strip()
    start_str   = request.GET.get('start', '').strip()
    end_str     = request.GET.get('end',   '').strip()

    today = date.today()

    # Parse custom date range; fall back to last 30 days
    if start_str and end_str:
        try:
            start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date   = datetime.strptime(end_str,   '%Y-%m-%d').date()
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            end_date = min(end_date, today)         # never beyond today
        except ValueError:
            end_date   = today
            start_date = today - timedelta(days=29)
    elif request.GET.get('days'):
        try:
            range_days = max(1, min(730, int(request.GET.get('days'))))
        except ValueError:
            range_days = 30
        end_date   = today
        start_date = today - timedelta(days=range_days - 1)
    else:
        end_date   = today
        start_date = today - timedelta(days=29)

    range_days = (end_date - start_date).days + 1

    base_qs = Incident.objects.filter(
        shop_code__in=shops,
        status__in=['Pending', 'Approved', 'Rejected'],
        submit_date__date__gte=start_date,
        submit_date__date__lte=end_date,
    )
    if dept_filter:
        base_qs = base_qs.filter(department=dept_filter)

    # ── Value totals ───────────────────────────────────────────
    total_wastage  = float(base_qs.aggregate(t=Sum('total_value'))['t'] or 0)
    approved_value = float(base_qs.filter(status='Approved').aggregate(t=Sum('total_value'))['t'] or 0)
    rejected_value = float(base_qs.filter(status='Rejected').aggregate(t=Sum('total_value'))['t'] or 0)
    pending_value  = float(base_qs.filter(status='Pending').aggregate(t=Sum('total_value'))['t'] or 0)
    today_wastage  = float(Incident.objects.filter(
        shop_code__in=shops, status__in=['Pending', 'Approved', 'Rejected'],
        submit_date__date=end_date,
    ).aggregate(t=Sum('total_value'))['t'] or 0)

    # ── Incident counts (one row per incident, not per item) ───
    inc_groups = {
        r['incident_no']: r['min_pk']
        for r in base_qs.values('incident_no').annotate(min_pk=Min('pk'))
    }
    rep_pks = list(inc_groups.values())
    rep_qs  = Incident.objects.filter(pk__in=rep_pks)
    pending_count  = rep_qs.filter(status='Pending').count()
    approved_count = rep_qs.filter(status='Approved').count()
    rejected_count = rep_qs.filter(status='Rejected').count()
    total_incidents = pending_count + approved_count + rejected_count
    closed = approved_count + rejected_count
    approval_rate = round(approved_count / closed * 100, 1) if closed > 0 else 0

    # ── Avg resolution time ────────────────────────────────────
    res_rows = list(
        base_qs.filter(status__in=['Approved', 'Rejected'], approved_date__isnull=False)
        .values('submit_date', 'approved_date')[:500]
    )
    res_hrs = [
        (r['approved_date'] - r['submit_date']).total_seconds() / 3600
        for r in res_rows if r['approved_date'] and r['submit_date']
    ]
    avg_resolution_hrs = round(sum(res_hrs) / len(res_hrs), 1) if res_hrs else 0

    # ── Week-over-week ─────────────────────────────────────────
    this_week_start = end_date - timedelta(days=6)
    last_week_end   = this_week_start - timedelta(days=1)
    last_week_start = last_week_end - timedelta(days=6)
    _wf = dict(shop_code__in=shops, status__in=['Pending', 'Approved', 'Rejected'])
    this_week_val = float(Incident.objects.filter(
        **_wf, submit_date__date__gte=this_week_start, submit_date__date__lte=end_date,
    ).aggregate(t=Sum('total_value'))['t'] or 0)
    last_week_val = float(Incident.objects.filter(
        **_wf, submit_date__date__gte=last_week_start, submit_date__date__lte=last_week_end,
    ).aggregate(t=Sum('total_value'))['t'] or 0)
    wow_pct = round((this_week_val - last_week_val) / last_week_val * 100, 1) if last_week_val > 0 else None

    # ── Daily trend (fill zeros for missing days) ──────────────
    daily_qs = (
        base_qs.annotate(day=TruncDate('submit_date')).values('day')
        .annotate(total=Sum('total_value'), cnt=Count('incident_no', distinct=True))
        .order_by('day')
    )
    d_map = {r['day']: (float(r['total']), r['cnt']) for r in daily_qs}
    trend_labels, trend_values, trend_counts = [], [], []
    for i in range(range_days):
        d = start_date + timedelta(days=i)
        tv, tc = d_map.get(d, (0, 0))
        trend_labels.append(d.strftime('%d %b'))
        trend_values.append(round(tv, 2))
        trend_counts.append(tc)

    # ── Breakdowns ─────────────────────────────────────────────
    by_reason = list(
        base_qs.values('reason')
        .annotate(total=Sum('total_value'), cnt=Count('pk'))
        .order_by('-total')
    )
    for r in by_reason:
        r['total'] = float(r['total'])

    by_dept = list(
        base_qs.values('department')
        .annotate(total=Sum('total_value'), incidents=Count('incident_no', distinct=True))
        .order_by('-total')
    )
    for r in by_dept:
        r['total'] = float(r['total'])

    by_cat = list(
        base_qs.values('category')
        .annotate(total=Sum('total_value'), cnt=Count('pk'))
        .order_by('-total')
    )
    for r in by_cat:
        r['total'] = float(r['total'])

    # ── Day-of-week pattern ────────────────────────────────────
    dow_qs = (
        base_qs.annotate(dow=ExtractWeekDay('submit_date')).values('dow')
        .annotate(total=Sum('total_value'), cnt=Count('incident_no', distinct=True))
        .order_by('dow')
    )
    # Django: 1=Sunday … 7=Saturday
    _DOW = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
    dow_map = {r['dow']: float(r['total']) for r in dow_qs}
    dow_labels = [_DOW[i] for i in range(1, 8)]
    dow_values = [dow_map.get(i, 0) for i in range(1, 8)]
    peak_dow = _DOW[max(range(1, 8), key=lambda i: dow_map.get(i, 0))] if any(dow_values) else None

    # ── Top 10 items ───────────────────────────────────────────
    top_items = list(
        base_qs.values('item_code', 'item_name')
        .annotate(total=Sum('total_value'), cnt=Count('pk'), qty=Sum('quantity'))
        .order_by('-total')[:10]
    )
    for r in top_items:
        r['total'] = float(r['total'])
        r['qty']   = float(r['qty'])
    top10_total = sum(r['total'] for r in top_items)
    top10_pct   = round(top10_total / total_wastage * 100) if total_wastage > 0 else 0

    # ── Dept: approved vs rejected value ──────────────────────
    dept_ar_qs = list(
        base_qs.filter(status__in=['Approved', 'Rejected'])
        .values('department', 'status')
        .annotate(total=Sum('total_value'))
        .order_by('department')
    )
    all_ar_depts = sorted(set(r['department'] for r in dept_ar_qs))
    dept_ar_approved = [float(next((r['total'] for r in dept_ar_qs if r['department'] == d and r['status'] == 'Approved'), 0)) for d in all_ar_depts]
    dept_ar_rejected = [float(next((r['total'] for r in dept_ar_qs if r['department'] == d and r['status'] == 'Rejected'), 0)) for d in all_ar_depts]

    # ── Recurring waste (same item ≥3 in period) ──────────────
    recurring = list(
        base_qs.values('item_code', 'item_name', 'reason')
        .annotate(freq=Count('pk'), total=Sum('total_value'))
        .filter(freq__gte=3)
        .order_by('-freq')[:10]
    )
    for r in recurring:
        r['total'] = float(r['total'])

    # ── Pending > 24 hours ────────────────────────────────────
    cutoff_24h = timezone.now() - timedelta(hours=24)
    now_ts     = timezone.now()
    old_pending = [
        {
            'incident_no': r['incident_no'],
            'pk': r['min_pk'],
            'hours': round((now_ts - r['submit_ts']).total_seconds() / 3600, 1),
            'total': float(r['total']),
            'items': r['items'],
            'submit_ts': r['submit_ts'],
        }
        for r in Incident.objects.filter(
            shop_code__in=shops, status='Pending', submit_date__lt=cutoff_24h,
        ).values('incident_no').annotate(
            min_pk=Min('pk'), submit_ts=Min('submit_date'),
            total=Sum('total_value'), items=Count('pk'),
        ).order_by('submit_ts')[:25]
    ]

    # ── Zero-submission days (missing daily reports) ──────────
    days_with_data = set(
        Incident.objects.filter(
            shop_code__in=shops, status__in=['Pending', 'Approved', 'Rejected'],
            submit_date__date__gte=start_date, submit_date__date__lt=end_date,
        ).annotate(day=TruncDate('submit_date')).values_list('day', flat=True).distinct()
    )
    zero_days = [
        start_date + timedelta(days=i)
        for i in range(range_days - 1)
        if (start_date + timedelta(days=i)) not in days_with_data
    ]

    # ── Dynamic insight text ───────────────────────────────────
    top_reason = by_reason[0]['reason'] if by_reason else None
    top_reason_pct = round(by_reason[0]['total'] / total_wastage * 100) if total_wastage > 0 and by_reason else 0
    top_dept = by_dept[0]['department'] if by_dept else None

    ctx = {
        'staff': staff, 'today': today, 'start_date': start_date,
        'end_date': end_date, 'range_days': range_days, 'dept_filter': dept_filter,
        'start_date_str': start_date.strftime('%Y-%m-%d'),
        'end_date_str':   end_date.strftime('%Y-%m-%d'),
        'today_str':      today.strftime('%Y-%m-%d'),
        'departments': Incident.DEPARTMENT_CHOICES, 'shops': shops,
        # KPIs
        'total_wastage': total_wastage, 'today_wastage': today_wastage,
        'total_incidents': total_incidents, 'pending_count': pending_count,
        'approved_count': approved_count, 'rejected_count': rejected_count,
        'approval_rate': approval_rate, 'avg_resolution_hrs': avg_resolution_hrs,
        'approved_value': approved_value, 'rejected_value': rejected_value,
        'pending_value': pending_value, 'wow_pct': wow_pct,
        'this_week_val': this_week_val,
        # Insight callouts
        'top_reason': top_reason, 'top_reason_pct': top_reason_pct,
        'top_dept': top_dept, 'top10_pct': top10_pct, 'peak_dow': peak_dow,
        # Chart JSON
        'trend_labels_j':    json.dumps(trend_labels),
        'trend_values_j':    json.dumps(trend_values),
        'trend_counts_j':    json.dumps(trend_counts),
        'reason_labels_j':   json.dumps([r['reason'] for r in by_reason]),
        'reason_values_j':   json.dumps([r['total'] for r in by_reason]),
        'dept_labels_j':     json.dumps([r['department'] for r in by_dept]),
        'dept_values_j':     json.dumps([r['total'] for r in by_dept]),
        'cat_labels_j':      json.dumps([r['category'] for r in by_cat]),
        'cat_values_j':      json.dumps([r['total'] for r in by_cat]),
        'dow_labels_j':      json.dumps(dow_labels),
        'dow_values_j':      json.dumps(dow_values),
        'top_item_names_j':  json.dumps([r['item_name'][:30] for r in top_items]),
        'top_item_values_j': json.dumps([r['total'] for r in top_items]),
        'dept_ar_labels_j':   json.dumps(all_ar_depts),
        'dept_ar_approved_j': json.dumps(dept_ar_approved),
        'dept_ar_rejected_j': json.dumps(dept_ar_rejected),
        'has_dept_ar': bool(all_ar_depts),
        # Tables
        'by_reason': by_reason, 'by_dept': by_dept, 'by_cat': by_cat,
        'top_items': top_items, 'old_pending': old_pending,
        'recurring': recurring, 'zero_days': zero_days,
    }
    return render(request, 'wastage/analytics_dashboard.html', ctx)


@login_required
def analytics_export_top_items(request):
    """
    CSV download: ALL wastage items for the selected date range.
    Columns: #, Item Code, Item Name, Incidents, Qty Wasted, Value (GH₵),
             Department, Shop Code, Approver Name,
             Total Pending (GH₵), Total Approved (GH₵)
    """
    import csv as csv_mod
    from collections import defaultdict
    from django.http import HttpResponse as _HR

    staff = get_staff_profile(request.user)
    if not staff or not _is_management(request.user):
        return _HR("Access denied", status=403)

    shops = get_managed_shops(staff)

    dept_filter = request.GET.get('dept', '').strip()
    start_str   = request.GET.get('start', '').strip()
    end_str     = request.GET.get('end',   '').strip()
    today       = date.today()

    if start_str and end_str:
        try:
            start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date   = datetime.strptime(end_str,   '%Y-%m-%d').date()
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            end_date = min(end_date, today)
        except ValueError:
            end_date   = today
            start_date = today - timedelta(days=29)
    else:
        end_date   = today
        start_date = today - timedelta(days=29)

    base_qs = Incident.objects.filter(
        shop_code__in=shops,
        status__in=['Pending', 'Approved', 'Rejected'],
        submit_date__date__gte=start_date,
        submit_date__date__lte=end_date,
    )
    if dept_filter:
        base_qs = base_qs.filter(department=dept_filter)

    # Fetch all detail rows in one query, group in Python
    rows = list(base_qs.values(
        'item_code', 'item_name', 'department', 'shop_code',
        'approved_name', 'status', 'total_value', 'quantity',
    ))

    # Group by (item_code, item_name) — preserving the name seen for each code
    _buckets = defaultdict(lambda: {
        'cnt': 0,
        'qty': 0.0,
        'total': 0.0,
        'departments': set(),
        'shop_codes': set(),
        'approver_names': set(),
        'total_pending': 0.0,
        'total_approved': 0.0,
    })
    _names = {}   # item_code → item_name (first seen)

    for row in rows:
        code = row['item_code'] or ''
        name = row['item_name'] or ''
        _names.setdefault(code, name)
        b = _buckets[code]
        b['cnt']   += 1
        b['qty']   += float(row['quantity']    or 0)
        b['total'] += float(row['total_value'] or 0)
        dept = (row['department'] or '').strip()
        if dept:
            b['departments'].add(dept)
        shop = (row['shop_code'] or '').strip()
        if shop:
            b['shop_codes'].add(shop)
        approver = (row['approved_name'] or '').strip()
        if approver:
            b['approver_names'].add(approver)
        if row['status'] == 'Pending':
            b['total_pending']  += float(row['total_value'] or 0)
        elif row['status'] == 'Approved':
            b['total_approved'] += float(row['total_value'] or 0)

    all_items = sorted(
        [{'item_code': code, 'item_name': _names[code], **data}
         for code, data in _buckets.items()],
        key=lambda x: -x['total'],
    )

    filename = (
        f"wastage_items_{start_date.strftime('%d%b%Y')}"
        f"_to_{end_date.strftime('%d%b%Y')}.csv"
    )
    response = _HR(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'

    writer = csv_mod.writer(response)
    writer.writerow([
        f'Wastage Items Report: {start_date.strftime("%d %b %Y")} '
        f'to {end_date.strftime("%d %b %Y")}'
    ])
    if dept_filter:
        writer.writerow([f'Department Filter: {dept_filter}'])
    writer.writerow([])
    writer.writerow([
        '#', 'Item Code', 'Item Name',
        'Incidents', 'Qty Wasted', 'Value (GH₵)',
        'Department', 'Shop Code', 'Approver Name',
        'Total Pending (GH₵)', 'Total Approved (GH₵)',
    ])

    for i, r in enumerate(all_items, 1):
        writer.writerow([
            i,
            r['item_code'],
            r['item_name'],
            r['cnt'],
            f"{r['qty']:.2f}",
            f"{r['total']:.2f}",
            ', '.join(sorted(r['departments'])),
            ', '.join(sorted(r['shop_codes'])),
            ', '.join(sorted(r['approver_names'])),
            f"{r['total_pending']:.2f}",
            f"{r['total_approved']:.2f}",
        ])

    return response


# ─────────────────────────────────────────────────────────────
# USER MANAGEMENT (manager-only)
# ─────────────────────────────────────────────────────────────

def _is_manager(staff):
    """True if staff is a Manager role OR a Django superuser."""
    return staff and (
        staff.role == 'manager' or
        (staff.user and staff.user.is_superuser)
    )


@login_required
def staff_list(request):
    current_staff = get_staff_profile(request.user)
    if not _is_manager(current_staff):
        messages.error(request, "User management is restricted to Managers.")
        return redirect('dashboard')

    q           = request.GET.get('q', '').strip()
    role_filter = request.GET.get('role', '')
    status_flt  = request.GET.get('status', '')

    qs = ShopStaff.objects.select_related('user').order_by('shop_code', 'emp_name')

    # Superusers see everyone; managers see only their shops
    if not (current_staff.user and current_staff.user.is_superuser):
        shops = get_managed_shops(current_staff)
        qs = qs.filter(shop_code__in=shops)

    if q:
        qs = qs.filter(
            Q(emp_id__icontains=q) | Q(emp_name__icontains=q) |
            Q(shop_code__icontains=q) | Q(shop_name__icontains=q)
        )
    if role_filter:
        qs = qs.filter(role=role_filter)
    if status_flt == 'active':
        qs = qs.filter(is_active=True)
    elif status_flt == 'inactive':
        qs = qs.filter(is_active=False)

    total       = qs.count()
    active_cnt  = qs.filter(is_active=True).count()
    approver_cnt = qs.filter(role__in=['supervisor', 'manager']).count()

    return render(request, 'wastage/staff_list.html', {
        'current_staff': current_staff,
        'staff_list': qs,
        'q': q, 'role_filter': role_filter, 'status_flt': status_flt,
        'today': date.today(),
        'role_choices': ShopStaff.ROLE_CHOICES,
        'total': total, 'active_cnt': active_cnt, 'approver_cnt': approver_cnt,
    })


@login_required
def staff_create(request):
    current_staff = get_staff_profile(request.user)
    if not _is_manager(current_staff):
        messages.error(request, "User management is restricted to Managers.")
        return redirect('dashboard')

    errors = []
    if request.method == 'POST':
        emp_id        = request.POST.get('emp_id', '').strip().upper()
        emp_name      = request.POST.get('emp_name', '').strip()
        shop_code     = request.POST.get('shop_code', '').strip().upper()
        shop_name     = request.POST.get('shop_name', '').strip()
        department    = request.POST.get('department', '').strip()
        role          = request.POST.get('role', 'staff').strip()
        managed_shops = request.POST.get('managed_shops', '').strip()
        approver_name = request.POST.get('approver_name', '').strip()
        is_active     = request.POST.get('is_active', '') == 'on'
        password      = request.POST.get('password', '')
        confirm_pw    = request.POST.get('confirm_password', '')

        if not emp_id:
            errors.append("Employee ID is required.")
        elif ShopStaff.objects.filter(emp_id=emp_id).exists():
            errors.append(f"Employee ID '{emp_id}' already exists.")
        if not emp_name:
            errors.append("Employee Name is required.")
        if not shop_code:
            errors.append("Shop Code is required.")
        if not shop_name:
            errors.append("Shop Name is required.")
        if role not in [r[0] for r in ShopStaff.ROLE_CHOICES]:
            errors.append("Invalid role.")
        if not password:
            errors.append("Password is required.")
        elif len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        elif password != confirm_pw:
            errors.append("Passwords do not match.")

        if not errors:
            # Build a unique Django username
            base_uname = emp_id.lower()
            uname = base_uname
            suffix = 1
            while User.objects.filter(username=uname).exists():
                uname = f"{base_uname}_{suffix}"
                suffix += 1

            user = User.objects.create_user(
                username=uname,
                password=password,
                first_name=emp_name.split()[0],
                last_name=' '.join(emp_name.split()[1:]),
                is_active=is_active,
            )
            ShopStaff.objects.create(
                user=user,
                emp_id=emp_id,
                emp_name=emp_name,
                shop_code=shop_code,
                shop_name=shop_name,
                department=department or None,
                role=role,
                managed_shops=managed_shops or None,
                approver_name=approver_name or None,
                is_active=is_active,
            )
            role_label = dict(ShopStaff.ROLE_CHOICES).get(role, role)
            messages.success(request,
                f"Login created for <strong>{emp_name}</strong> ({emp_id}) — {role_label}.")
            return redirect('staff_list')

    return render(request, 'wastage/staff_form.html', {
        'current_staff': current_staff,
        'errors': errors,
        'post': request.POST,
        'edit_mode': False,
        'role_choices': ShopStaff.ROLE_CHOICES,
        'dept_choices': Incident.DEPARTMENT_CHOICES,
        'today': date.today(),
    })


@login_required
def staff_edit(request, pk):
    current_staff = get_staff_profile(request.user)
    if not _is_manager(current_staff):
        messages.error(request, "User management is restricted to Managers.")
        return redirect('dashboard')

    target = get_object_or_404(ShopStaff, pk=pk)
    errors = []

    if request.method == 'POST':
        emp_name      = request.POST.get('emp_name', '').strip()
        shop_code     = request.POST.get('shop_code', '').strip().upper()
        shop_name     = request.POST.get('shop_name', '').strip()
        department    = request.POST.get('department', '').strip()
        role          = request.POST.get('role', target.role).strip()
        managed_shops = request.POST.get('managed_shops', '').strip()
        approver_name = request.POST.get('approver_name', '').strip()
        is_active     = request.POST.get('is_active', '') == 'on'

        if not emp_name:
            errors.append("Employee Name is required.")
        if not shop_code:
            errors.append("Shop Code is required.")
        if not shop_name:
            errors.append("Shop Name is required.")
        if role not in [r[0] for r in ShopStaff.ROLE_CHOICES]:
            errors.append("Invalid role.")

        if not errors:
            target.emp_name      = emp_name
            target.shop_code     = shop_code
            target.shop_name     = shop_name
            target.department    = department or None
            target.role          = role
            target.managed_shops = managed_shops or None
            target.approver_name = approver_name or None
            target.is_active     = is_active
            target.save()
            if target.user:
                target.user.first_name = emp_name.split()[0]
                target.user.last_name  = ' '.join(emp_name.split()[1:])
                target.user.is_active  = is_active
                target.user.save(update_fields=['first_name', 'last_name', 'is_active'])
            messages.success(request, f"Profile updated for <strong>{target.emp_name}</strong>.")
            return redirect('staff_list')

    # Pre-fill with existing values on GET
    post_data = request.POST if request.method == 'POST' else {
        'emp_name': target.emp_name, 'shop_code': target.shop_code,
        'shop_name': target.shop_name, 'department': target.department or '',
        'role': target.role, 'managed_shops': target.managed_shops or '',
        'approver_name': target.approver_name or '',
        'is_active': 'on' if target.is_active else '',
    }
    return render(request, 'wastage/staff_form.html', {
        'current_staff': current_staff,
        'target': target,
        'errors': errors,
        'post': post_data,
        'edit_mode': True,
        'role_choices': ShopStaff.ROLE_CHOICES,
        'dept_choices': Incident.DEPARTMENT_CHOICES,
        'today': date.today(),
    })


@login_required
@require_POST
def staff_toggle_active(request, pk):
    current_staff = get_staff_profile(request.user)
    if not _is_manager(current_staff):
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    target = get_object_or_404(ShopStaff, pk=pk)
    # Prevent disabling own account
    if target.user and target.user == request.user:
        return JsonResponse({'error': 'You cannot disable your own account.'}, status=400)

    target.is_active = not target.is_active
    target.save(update_fields=['is_active'])
    if target.user:
        target.user.is_active = target.is_active
        target.user.save(update_fields=['is_active'])

    return JsonResponse({
        'success': True,
        'is_active': target.is_active,
        'label': 'Active' if target.is_active else 'Inactive',
    })


@login_required
@require_POST
def staff_reset_password(request, pk):
    current_staff = get_staff_profile(request.user)
    if not _is_manager(current_staff):
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    target = get_object_or_404(ShopStaff, pk=pk)

    try:
        body       = json.loads(request.body)
        new_pw     = body.get('password', '').strip()
        confirm_pw = body.get('confirm', '').strip()
    except Exception:
        return JsonResponse({'error': 'Invalid request.'}, status=400)

    if not new_pw:
        return JsonResponse({'error': 'Password is required.'}, status=400)
    if len(new_pw) < 6:
        return JsonResponse({'error': 'Password must be at least 6 characters.'}, status=400)
    if new_pw != confirm_pw:
        return JsonResponse({'error': 'Passwords do not match.'}, status=400)
    if not target.user:
        return JsonResponse({'error': 'No user account linked to this staff record.'}, status=400)

    target.user.set_password(new_pw)
    target.user.save()
    return JsonResponse({'success': True, 'message': f'Password reset for {target.emp_name}.'})


# ─────────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────────

@login_required
def reports(request):
    staff = get_staff_profile(request.user)
    if not staff or (not staff.is_approver and not _is_management(request.user)):
        return redirect('dashboard')

    return render(request, 'wastage/reports.html', {
        'staff': staff,
        'today': date.today(),
    })


@login_required
def daily_report(request):
    staff = get_staff_profile(request.user)
    if not staff or (not staff.is_approver and not _is_management(request.user)):
        return redirect('dashboard')

    report_date_str = request.GET.get('date', str(date.today()))
    try:
        report_date = datetime.strptime(report_date_str, '%Y-%m-%d').date()
    except ValueError:
        report_date = date.today()

    # Management sees all shops; supervisors see only their managed shops
    if _is_management(request.user):
        incidents = Incident.objects.filter(
            submit_date__date=report_date
        ).order_by('department', '-total_value')
    else:
        incidents = Incident.objects.filter(
            shop_code__in=get_managed_shops(staff),
            submit_date__date=report_date
        ).order_by('department', '-total_value')

    summary = incidents.values('department').annotate(
        count=Count('incident_no', distinct=True),
        total=Sum('total_value')
    ).order_by('department')

    grand_total = incidents.aggregate(total=Sum('total_value'))['total'] or 0
    by_reason = incidents.values('reason').annotate(
        count=Count('incident_no', distinct=True),
        total=Sum('total_value')
    ).order_by('-total')

    return render(request, 'wastage/daily_report.html', {
        'staff':        staff,
        'report_date':  report_date,
        'incidents':    incidents,
        'summary':      summary,
        'grand_total':  grand_total,
        'by_reason':    by_reason,
        'today':        date.today(),
        'is_mgmt_view': _is_management(request.user),
    })


# ─────────────────────────────────────────────────────────────
# AJAX APIs
# ─────────────────────────────────────────────────────────────

@login_required
@require_GET
def item_search_api(request):
    """Return matching items from ItemMaster as JSON."""
    q = request.GET.get('q', '').strip()
    if len(q) < 2:
        return JsonResponse({'items': []})

    items = ItemMaster.objects.filter(
        Q(item_name__icontains=q) | Q(item_code__icontains=q),
        is_active=True
    ).values('item_code', 'item_name', 'category', 'selling_price', 'uom')[:20]

    return JsonResponse({'items': list(items)})


@login_required
@require_GET
def item_detail_api(request, item_code):
    """Return single item detail as JSON."""
    try:
        item = ItemMaster.objects.get(item_code=item_code, is_active=True)
        return JsonResponse({
            'item_code': item.item_code,
            'item_name': item.item_name,
            'category': item.category,
            'selling_price': str(item.selling_price),
            'uom': item.uom or '',
        })
    except ItemMaster.DoesNotExist:
        return JsonResponse({'error': 'Item not found'}, status=404)


# ─────────────────────────────────────────────────────────────
# PWA
# ─────────────────────────────────────────────────────────────

def pwa_manifest(request):
    manifest = {
        "name": "Melcom Cost Control Portal",
        "short_name": "CostControl",
        "description": "Wastage & Incident Management System",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#1a1a2e",
        "theme_color": "#e31837",
        "orientation": "portrait-primary",
        "lang": "en",
        "icons": [
            {"src": "/static/img/icon-72.png",  "sizes": "72x72",   "type": "image/png", "purpose": "any"},
            {"src": "/static/img/icon-96.png",  "sizes": "96x96",   "type": "image/png", "purpose": "any"},
            {"src": "/static/img/icon-128.png", "sizes": "128x128", "type": "image/png", "purpose": "any"},
            {"src": "/static/img/icon-144.png", "sizes": "144x144", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/img/icon-152.png", "sizes": "152x152", "type": "image/png", "purpose": "any"},
            {"src": "/static/img/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/img/icon-384.png", "sizes": "384x384", "type": "image/png", "purpose": "any"},
            {"src": "/static/img/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
        "screenshots": [
            {
                "src": "/static/img/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "form_factor": "narrow",
                "label": "Melcom Cost Control Portal"
            }
        ],
        "categories": ["productivity", "business"],
        "prefer_related_applications": False,
    }
    return HttpResponse(
        json.dumps(manifest),
        content_type='application/manifest+json'
    )


def service_worker(request):
    sw_content = """
const CACHE_NAME = 'melcom-ccp-v4';
const OFFLINE_URL = '/offline/';

// Core shell — always cached at install
const SHELL_ASSETS = [
  '/login/',
  '/offline/',
  '/static/css/main.css',
  '/static/img/icon-192.png',
  '/static/img/icon-512.png',
];

// ── Install: pre-cache shell ──────────────────────────────────────────────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(SHELL_ASSETS);
    })
  );
  self.skipWaiting();
});

// ── Activate: purge old caches ────────────────────────────────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch: network-first with offline fallback ────────────────────────────────
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Static assets: cache-first (CSS, images, fonts)
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // API calls: always network, no caching
  if (url.pathname.startsWith('/api/') || url.pathname.includes('/upload-media/') ||
      url.pathname.includes('/approve/') || url.pathname.includes('/reject/')) {
    return;
  }

  // Pages: network-first, fall back to offline page
  event.respondWith(
    fetch(event.request).catch(() =>
      caches.match(event.request).then(r => r || caches.match(OFFLINE_URL))
    )
  );
});

// ── Background sync placeholder ───────────────────────────────────────────────
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-incidents') {
    // Future: retry failed submissions when back online
    console.log('[SW] Background sync triggered');
  }
});
"""
    return HttpResponse(sw_content, content_type='application/javascript')


def offline_view(request):
    return render(request, 'wastage/offline.html')


# ─────────────────────────────────────────────────────────────
# USER MANAGEMENT  (requires separate MGMT authentication)
# ─────────────────────────────────────────────────────────────

def _um_auth_required(request):
    """Returns True if the request has a valid MGMT user-management session."""
    return request.session.get('um_authenticated') is True


def um_login(request):
    """Separate login gate for User Management — only MGMT (is_staff) users allowed."""
    next_url = request.GET.get('next') or request.POST.get('next') or '/user-management/'
    error = None

    if request.method == 'POST':
        emp_id   = request.POST.get('emp_id', '').strip()
        password = request.POST.get('password', '').strip()

        try:
            staff = ShopStaff.objects.get(emp_id=emp_id, is_active=True)
        except ShopStaff.DoesNotExist:
            error = "Invalid credentials."
        else:
            if not staff.user:
                error = "Account not configured. Contact IT."
            else:
                user = authenticate(request, username=staff.user.username, password=password)
                if user and _is_management(user):
                    request.session['um_authenticated'] = True
                    request.session['um_emp_id']        = emp_id
                    return redirect(next_url)
                elif user:
                    error = "You do not have Management access."
                else:
                    error = "Invalid credentials."

    return render(request, 'wastage/um_login.html', {'error': error, 'next': next_url})


def um_logout(request):
    request.session.pop('um_authenticated', None)
    request.session.pop('um_emp_id', None)
    return redirect('/login/')


def um_user_list(request):
    if not _um_auth_required(request):
        return redirect(f'/user-management/login/?next=/user-management/')

    users        = ShopStaff.objects.select_related('user').order_by('emp_name')
    active_count = users.filter(is_active=True).count()

    return render(request, 'wastage/user_management.html', {
        'users':          users,
        'active_count':   active_count,
        'inactive_count': users.count() - active_count,
    })


def um_create_user(request):
    if not _um_auth_required(request):
        return redirect('/user-management/login/?next=/user-management/')

    if request.method != 'POST':
        return redirect('/user-management/')

    emp_id     = request.POST.get('emp_id', '').strip().upper()
    emp_name   = request.POST.get('emp_name', '').strip()
    shop_code  = request.POST.get('shop_code', '').strip().upper()
    shop_name  = request.POST.get('shop_name', '').strip()
    role       = request.POST.get('role', 'staff').strip()
    department = request.POST.get('department', '').strip()
    password   = request.POST.get('password', '').strip()

    if not all([emp_id, emp_name, shop_code, shop_name, password]):
        messages.error(request, "All required fields must be filled.")
        return redirect('/user-management/')

    if ShopStaff.objects.filter(emp_id=emp_id).exists():
        messages.error(request, f"Employee ID '{emp_id}' already exists.")
        return redirect('/user-management/')

    if len(password) < 6:
        messages.error(request, "Password must be at least 6 characters.")
        return redirect('/user-management/')

    # Create Django user — Management role gets is_staff=True (superuser access)
    username = emp_id.lower()
    if User.objects.filter(username=username).exists():
        username = f"{emp_id.lower()}_{shop_code.lower()}"

    is_mgmt = (role == 'management')
    dj_user = User.objects.create_user(
        username=username, password=password,
        first_name=emp_name, is_active=True,
        is_staff=is_mgmt,
    )

    # Management users automatically get all shops in managed_shops
    all_shops_str = ''
    if is_mgmt:
        all_shop_codes = list(
            ShopStaff.objects.values_list('shop_code', flat=True).distinct()
        )
        all_shops_str = ','.join(s.upper() for s in all_shop_codes if s)

    ShopStaff.objects.create(
        emp_id=emp_id, emp_name=emp_name,
        shop_code=shop_code, shop_name=shop_name,
        role=role, department=department or None,
        user=dj_user, is_active=True,
        managed_shops=all_shops_str if is_mgmt else '',
        managed_shop_name='All Shops' if is_mgmt else '',
    )

    messages.success(request, f"User '{emp_name}' ({emp_id}) created successfully.")
    return redirect('/user-management/')


def um_edit_user(request, emp_id):
    if not _um_auth_required(request):
        return redirect('/user-management/login/?next=/user-management/')

    staff = get_object_or_404(ShopStaff, emp_id=emp_id)

    if request.method != 'POST':
        return redirect('/user-management/')

    new_role = request.POST.get('role', staff.role).strip()
    staff.emp_name   = request.POST.get('emp_name', staff.emp_name).strip()
    staff.shop_code  = request.POST.get('shop_code', staff.shop_code).strip().upper()
    staff.shop_name  = request.POST.get('shop_name', staff.shop_name).strip()
    staff.role       = new_role
    staff.department = request.POST.get('department', '').strip() or None
    staff.is_active  = request.POST.get('is_active', 'true') == 'true'

    # Management role always owns all shops
    if new_role == 'management':
        all_shop_codes = list(
            ShopStaff.objects.exclude(emp_id=emp_id)
            .values_list('shop_code', flat=True).distinct()
        )
        staff.managed_shops     = ','.join(s.upper() for s in all_shop_codes if s)
        staff.managed_shop_name = 'All Shops'

    staff.save()

    # Sync Django user flags
    if staff.user:
        staff.user.first_name = staff.emp_name
        staff.user.is_active  = staff.is_active
        staff.user.is_staff   = (new_role == 'management')   # management = portal superuser
        staff.user.save()

    # Optional password reset
    new_password = request.POST.get('password', '').strip()
    if new_password:
        if len(new_password) < 6:
            messages.error(request, "Password must be at least 6 characters.")
            return redirect('/user-management/')
        if staff.user:
            staff.user.set_password(new_password)
            staff.user.save()

    messages.success(request, f"User '{staff.emp_name}' updated successfully.")
    return redirect('/user-management/')


def um_toggle_user(request, emp_id):
    if not _um_auth_required(request):
        return redirect('/user-management/login/?next=/user-management/')

    if ShopStaff.objects.filter(emp_id=emp_id, role='management').exists():
        messages.error(request, "Management accounts cannot be deactivated here. Edit the user to change their role first.")
        return redirect('/user-management/')

    staff = get_object_or_404(ShopStaff, emp_id=emp_id)
    staff.is_active = not staff.is_active
    staff.save()
    if staff.user:
        staff.user.is_active = staff.is_active
        staff.user.save()

    status = "activated" if staff.is_active else "deactivated"
    messages.success(request, f"User '{staff.emp_name}' has been {status}.")
    return redirect('/user-management/')


def um_delete_user(request, emp_id):
    if not _um_auth_required(request):
        return redirect('/user-management/login/?next=/user-management/')

    if ShopStaff.objects.filter(emp_id=emp_id, role='management').exists():
        messages.error(request, "Management accounts cannot be deleted. Deactivate them instead.")
        return redirect('/user-management/')

    if request.method != 'POST':
        return redirect('/user-management/')

    staff = get_object_or_404(ShopStaff, emp_id=emp_id)
    name  = staff.emp_name
    if staff.user:
        staff.user.delete()   # cascades to ShopStaff via OneToOne
    else:
        staff.delete()

    messages.success(request, f"User '{name}' ({emp_id}) has been deleted.")
    return redirect('/user-management/')


# ─────────────────────────────────────────────────────────────
# MGMT MASTER VIEWS  (read-only, all shops, all incidents)
# ─────────────────────────────────────────────────────────────

def _is_management(user) -> bool:
    """True if user has management access — either Django is_staff OR role='management' in ShopStaff."""
    if user.is_staff:
        return True
    try:
        return ShopStaff.objects.get(user=user).role == 'management'
    except ShopStaff.DoesNotExist:
        return False


@login_required
def mgmt_dashboard(request):
    """Master dashboard for MGMT users — two tabs: Staff incidents & Supervisor review."""
    if not _is_management(request.user):
        return redirect('dashboard')

    staff = get_staff_profile(request.user)

    # ── Staff tab: all incidents submitted by role='staff' ────────────────────
    staff_incidents = (
        Incident.objects
        .filter(
            submitted_by__in=ShopStaff.objects.filter(role='staff').values_list('emp_id', flat=True)
        )
        .order_by('-submit_date')
        .select_related()
    )

    # ── Supervisor tab: all incidents regardless of status ────────────────────
    all_incidents = Incident.objects.all().order_by('-submit_date')

    # Summary counts
    total_staff      = staff_incidents.count()
    total_all        = all_incidents.count()
    pending_count    = all_incidents.filter(status='Pending').count()
    approved_count   = all_incidents.filter(status='Approved').count()
    rejected_count   = all_incidents.filter(status='Rejected').count()

    # Per-status querysets for supervisor tab
    pending_qs  = all_incidents.filter(status='Pending')
    approved_qs = all_incidents.filter(status='Approved')
    rejected_qs = all_incidents.filter(status='Rejected')

    selected_role  = request.session.get('selected_role', 'management')
    is_mgmt_view   = selected_role == 'management'
    sup_tab_label  = 'Management View' if is_mgmt_view else 'Supervisor / Manager View'

    return render(request, 'wastage/mgmt_dashboard.html', {
        'staff':           staff,
        'staff_incidents': staff_incidents,
        'pending_qs':      pending_qs,
        'approved_qs':     approved_qs,
        'rejected_qs':     rejected_qs,
        'total_staff':     total_staff,
        'total_all':       total_all,
        'pending_count':   pending_count,
        'approved_count':  approved_count,
        'rejected_count':  rejected_count,
        'sup_tab_label':   sup_tab_label,
        'is_mgmt_view':    is_mgmt_view,
    })


@login_required
def mgmt_incident_detail(request, pk):
    """Read-only incident detail for MGMT — shows all items + photo. No edit allowed."""
    if not _is_management(request.user):
        return redirect('dashboard')

    staff    = get_staff_profile(request.user)
    incident = get_object_or_404(Incident, pk=pk)
    items    = list(Incident.objects.filter(incident_no=incident.incident_no).order_by('pk'))
    total    = sum(i.total_value for i in items)

    # Build photo URL if photo exists (reuse serve_media endpoint)
    photo_url = None
    if incident.photo_path:
        photo_url = f'/approvals/{pk}/media/photo/'

    return render(request, 'wastage/mgmt_incident_detail.html', {
        'staff':     staff,
        'incident':  incident,
        'items':     items,
        'total':     total,
        'photo_url': photo_url,
    })


@login_required
def mgmt_serve_media(request, pk, media_type):
    """Serve photo/video for MGMT detail view — bypasses shop restriction."""
    if not _is_management(request.user):
        return HttpResponse(status=403)

    incident = get_object_or_404(Incident, pk=pk)

    ALLOWED = {
        'photo': {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                  '.png': 'image/png',  '.webp': 'image/webp'},
        'video': {'.mp4': 'video/mp4',  '.mov': 'video/quicktime'},
    }
    if media_type not in ALLOWED:
        return HttpResponse(status=400)

    rel_path = incident.photo_path if media_type == 'photo' else incident.video_path
    if not rel_path:
        return HttpResponse(status=404)

    net_root  = getattr(settings, 'INCIDENT_MEDIA_ROOT', r'\\10.10.0.30\mis\wastagecontrol')
    full_path = os.path.join(net_root, rel_path)

    if not os.path.exists(full_path):
        return HttpResponse(status=404)

    ext       = os.path.splitext(full_path)[1].lower()
    mime      = ALLOWED[media_type].get(ext, 'application/octet-stream')

    with open(full_path, 'rb') as f:
        return HttpResponse(f.read(), content_type=mime)


# ─────────────────────────────────────────────────────────────
# AI ANALYST CHAT  (analytics dashboard bot widget)
# ─────────────────────────────────────────────────────────────

# Words that must never appear in an LLM-generated SQL query
_SQL_BANNED = {
    'DELETE', 'DROP', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER',
    'CREATE', 'REPLACE', 'GRANT', 'REVOKE', 'VACUUM', 'COPY',
    'EXECUTE', 'CALL', 'DO', 'MERGE',
}


def _ai_safe_sql(raw: str) -> str:
    """Strip markdown fences, verify SELECT-only, block destructive keywords."""
    # Remove code fences
    if raw.startswith('```'):
        raw = '\n'.join(l for l in raw.splitlines() if not l.startswith('```')).strip()

    # Must start with SELECT
    clean = raw.upper().lstrip()
    if not clean.startswith('SELECT'):
        raise ValueError('Non-SELECT query blocked.')

    # Block any destructive keyword (word-boundary match)
    import re as _re
    for kw in _SQL_BANNED:
        if _re.search(r'\b' + kw + r'\b', clean):
            raise ValueError(f'Keyword "{kw}" is not allowed.')

    # No semicolons (blocks stacked queries)
    if ';' in raw:
        raise ValueError('Semicolons not allowed in query.')

    return raw


def _call_llm(system_prompt: str, user_content: str, max_tokens: int = 600) -> str:
    """Call the configured LLM provider and return the text response."""
    import urllib.request as _urlreq

    provider = getattr(settings, 'AI_PROVIDER', 'ollama').lower()

    if provider == 'anthropic':
        api_key = getattr(settings, 'ANTHROPIC_API_KEY', '').strip()
        if not api_key or api_key == 'your-anthropic-api-key-here':
            raise RuntimeError('ANTHROPIC_API_KEY not set in settings.py.')
        try:
            import anthropic as _ant
        except ImportError:
            raise RuntimeError('Run: pip install anthropic')
        client = _ant.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_content}],
        )
        return resp.content[0].text.strip()

    # Default: Ollama (local, free)
    ollama_url = getattr(settings, 'OLLAMA_URL', 'http://localhost:11434').rstrip('/')
    model      = getattr(settings, 'OLLAMA_MODEL', 'llama3.1:8b')
    payload    = json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_content},
        ],
        'stream': False,
        'options': {'num_predict': max_tokens, 'temperature': 0.1},
    }).encode()
    req = _urlreq.Request(
        f'{ollama_url}/api/chat',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with _urlreq.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data['message']['content'].strip()
    except Exception as exc:
        raise RuntimeError(f'Ollama error: {exc}. Is Ollama running at {ollama_url}?')


@login_required
def analytics_ai_chat(request):
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': 'Method not allowed'}, status=405)

    try:
        body         = json.loads(request.body)
        user_message = body.get('message', '').strip()
        start_str    = body.get('start', '').strip()
        end_str      = body.get('end',   '').strip()
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({'ok': False, 'error': 'Invalid request'}, status=400)

    if not user_message:
        return JsonResponse({'ok': False, 'error': 'Empty message'}, status=400)

    try:
        staff = ShopStaff.objects.get(user=request.user)
    except ShopStaff.DoesNotExist:
        return JsonResponse({'ok': False, 'error': 'Staff profile not found'}, status=403)

    shops = get_managed_shops(staff)
    if not shops:
        return JsonResponse({'ok': False, 'error': 'No shops assigned'}, status=403)

    today_d = date.today()
    if not end_str:
        end_str = today_d.strftime('%Y-%m-%d')
    if not start_str:
        start_str = (today_d - timedelta(days=30)).strftime('%Y-%m-%d')

    shops_array = ', '.join(f"'{s}'" for s in shops)

    schema_context = f"""You are a PostgreSQL query generator for a food wastage cost control portal.

Database tables (Django-generated, PostgreSQL):

1. wastage_incident
   Columns: id, incident_no, shop_code, shop_name, department, submitted_by, submitted_name,
            item_code, item_name, category, quantity(numeric), uom, selling_price(numeric),
            total_value(numeric), reason, status, submit_date(timestamptz),
            approved_by, approved_name, approved_date(timestamptz),
            is_system_generated(boolean), is_late_submission(boolean)
   Values:  department IN ('Bakery','Kitchen','RM')
            category   IN ('RAW','Finished','Packaging','WIP','Semifinished')
            reason     IN ('Expired','Over Production','Damage','Quality Issue','Handling Error','No Wastage','Others')
            status     IN ('Pending','Approved','Rejected','Draft')

2. wastage_itemmaster
   Columns: id, item_code, item_name, department, grp, sub_group, uom, category,
            cost_price(numeric), selling_price(numeric), is_active(boolean)

3. wastage_shopstaff
   Columns: id, emp_id, emp_name, shop_code, shop_name, department, role

User scope:
  Shops: {shops_array}
  Date range: {start_str} to {end_str}

STRICT RULES — you MUST follow all of them:
1. Return ONLY a valid PostgreSQL SELECT statement — no markdown, no explanation, nothing else
2. ALWAYS include: shop_code = ANY(ARRAY[{shops_array}]) in the WHERE clause
3. Default date filter: submit_date::date BETWEEN '{start_str}' AND '{end_str}'
4. Exclude is_system_generated = TRUE unless the question is specifically about system records
5. Exclude Draft status: status != 'Draft'
6. For detail queries add LIMIT 20; no LIMIT for aggregations
7. Use SUM(total_value) for GHS amounts; COUNT(DISTINCT incident_no) for incident counts
8. You are ONLY allowed to read data — never write DELETE, UPDATE, INSERT, DROP, TRUNCATE or any other write operation"""

    insight_system = """You are a concise business analyst for a food cost control manager.
Convert the provided query results into a clear, direct insight answering the user's question.
Rules:
- 2–4 sentences maximum
- Include key figures (GHS amounts, %, counts) where relevant
- Use plain business language — no SQL terms, no table names, no "the data shows"
- Highlight the most important finding and any actionable implication
- If no data: say so briefly and suggest checking the date range or department filter
- Never display raw tables, lists, or JSON"""

    # Step 1: LLM generates SQL
    try:
        raw_sql = _call_llm(schema_context, user_message, max_tokens=600)
    except RuntimeError as exc:
        return JsonResponse({'ok': False, 'error': str(exc)}, status=503)

    try:
        safe_sql = _ai_safe_sql(raw_sql)
    except ValueError as exc:
        return JsonResponse({'ok': False, 'error': f'Query blocked: {exc}'}, status=400)

    # Step 2: Execute SQL (read-only, timeout-guarded)
    import decimal
    from django.db import connection
    db_results, db_error = [], None
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '8s'")
            cursor.execute(safe_sql)
            columns = [col[0] for col in cursor.description]
            rows    = cursor.fetchmany(20)
        for row in rows:
            rec = {}
            for k, v in zip(columns, row):
                if isinstance(v, decimal.Decimal):
                    rec[k] = float(v)
                elif hasattr(v, 'isoformat'):
                    rec[k] = v.isoformat()
                else:
                    rec[k] = v
            db_results.append(rec)
    except Exception as exc:
        db_error = str(exc)

    # Step 3: LLM converts results → insight
    if db_results:
        data_text = f"Query returned {len(db_results)} rows:\n{json.dumps(db_results, indent=2, default=str)}"
    elif db_error:
        data_text = f"The query failed: {db_error}"
    else:
        data_text = "The query returned no results for the specified filters and date range."

    try:
        insight = _call_llm(insight_system, f"Question: {user_message}\n\n{data_text}", max_tokens=350)
    except RuntimeError:
        insight = "Unable to generate an insight right now. Please try again."

    return JsonResponse({'ok': True, 'insight': insight})

