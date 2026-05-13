import base64
import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .claude_vision import analyse_image
from .item_matcher import match_product
from .models import ProductScan

logger = logging.getLogger(__name__)

# Roles allowed to access the scanner
ALLOWED_ROLES = {"supervisor", "management", "manager"}


def _get_profile(user):
    """Return ShopStaff profile or None."""
    try:
        from wastage.models import ShopStaff
        return ShopStaff.objects.get(user=user)
    except Exception:
        return None


def _can_access(user):
    """Return True if user has a role that may use the scanner."""
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser or user.is_staff:
        return True
    profile = _get_profile(user)
    if not profile:
        return False
    return profile.role in ALLOWED_ROLES


def scanner_page(request):
    if not _can_access(request.user):
        return redirect(f"/login/?next={request.path}")

    profile = _get_profile(request.user)
    recent  = ProductScan.objects.order_by("-scanned_at")[:100]
    return render(request, "product_scanner/scanner.html", {
        "staff":             profile,
        "recent_scans":      recent,
        "recent_scans_json": json.dumps([s.to_dict() for s in recent]),
    })




# ── Shared scan handler ──────────────────────────────────────────────────────

def _save_item(item: dict, raw_result: dict, request) -> dict:
    """Save one identified item to DB and run matcher. Returns to_dict() + item_matches."""

    def _f(key, default=None):
        v = item.get(key, default)
        return v if v not in ("", "null", "N/A", None) else default

    def _num(key, default=None):
        try:
            v = item.get(key)
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def _int(key, default=None):
        try:
            v = item.get(key)
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    scan = ProductScan.objects.create(
        product_name       = _f("product_name"),
        brand              = _f("brand"),
        category           = _f("category"),
        product_type       = _f("product_type"),
        description        = _f("description"),
        is_fnv             = item.get("is_fnv"),
        quantity           = _num("quantity"),
        unit_of_measurement= _f("unit_of_measurement"),
        units_per_case     = _int("units_per_case"),
        num_cases          = _int("num_cases"),
        num_pieces         = _int("num_pieces"),
        total_units        = _num("total_units"),
        grammage           = _f("grammage"),
        weight_per_unit    = _num("weight_per_unit"),
        weight_unit        = _f("weight_unit"),
        total_weight       = _num("total_weight"),
        is_alcohol         = bool(item.get("is_alcohol", False)),
        alcohol_type       = _f("alcohol_type"),
        alcohol_subtype    = _f("alcohol_subtype"),
        bottle_size_ml     = _num("bottle_size_ml"),
        fill_level_pct     = _num("fill_level_pct"),
        ml_remaining       = _num("ml_remaining"),
        ml_consumed        = _num("ml_consumed"),
        alcohol_percentage = _num("alcohol_percentage"),
        barcode            = _f("barcode"),
        expiry_date        = _f("expiry_date"),
        has_expiry         = item.get("has_expiry"),
        country_of_origin  = _f("country_of_origin"),
        scale_reading      = _f("scale_reading"),
        confidence_score   = _num("confidence"),
        ai_notes           = _f("notes"),
        raw_ai_response    = raw_result,
        model_used         = raw_result.get("_model", ""),
        session_key        = request.session.session_key or "",
        device_info        = request.META.get("HTTP_USER_AGENT", "")[:500],
    )

    # Match: include grammage in keywords for size-aware scoring
    _kw = list(item.get("search_keywords") or [])
    _gram = _f("grammage")
    if _gram and _gram not in _kw:
        _kw.append(_gram)
    matches = match_product(
        product_name=_f("product_name") or "",
        brand=_f("brand") or "",
        keywords=_kw,
    )

    data = scan.to_dict()
    data["item_matches"] = matches
    return data


def _process_scan(request, image_bytes: bytes, mime_type: str = "image/jpeg") -> JsonResponse:
    result = analyse_image(image_bytes, mime_type=mime_type)
    items  = result.get("items", [])

    if not items:
        return JsonResponse({"error": "AI returned no items"}, status=400)

    saved = [_save_item(item, result, request) for item in items]

    provider = result.get("_provider", "")
    model    = result.get("_model", "")
    fallback = result.get("_fallback", "")

    return JsonResponse({
        "success":           True,
        "multi":             len(saved) > 1,
        "item_count":        len(saved),
        "scene_description": result.get("scene_description", ""),
        "provider":          provider,
        "model":             model,
        "fallback_note":     fallback,
        "items":             saved,
        # backward-compat: single item as 'data'
        "data":              saved[0] if saved else {},
    })


# ── API: camera capture (base64) ─────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def api_scan(request):
    if not _can_access(request.user):
        return JsonResponse({"error": "Access denied"}, status=403)
    try:
        body     = json.loads(request.body)
        img_data = body.get("image", "")
        if not img_data:
            return JsonResponse({"error": "No image provided"}, status=400)

        if "," in img_data:
            header, b64str = img_data.split(",", 1)
            mime = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
        else:
            b64str = img_data
            mime   = "image/jpeg"

        return _process_scan(request, base64.b64decode(b64str), mime)
    except Exception as exc:
        logger.exception("Camera scan error: %s", exc)
        return JsonResponse({"error": str(exc)}, status=500)


# ── API: image file upload ───────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def api_scan_upload(request):
    if not _can_access(request.user):
        return JsonResponse({"error": "Access denied"}, status=403)
    try:
        uploaded = request.FILES.get("image")
        if not uploaded:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        mime = uploaded.content_type or "image/jpeg"
        image_bytes = uploaded.read()
        return _process_scan(request, image_bytes, mime)
    except Exception as exc:
        logger.exception("Upload scan error: %s", exc)
        return JsonResponse({"error": str(exc)}, status=500)


# ── API: history ─────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def api_history(request):
    if not _can_access(request.user):
        return JsonResponse({"error": "Access denied"}, status=403)
    scans = ProductScan.objects.order_by("-scanned_at")[:100]
    return JsonResponse({"scans": [s.to_dict() for s in scans]})


# ── API: delete ───────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["DELETE"])
def api_delete_scan(request, scan_id):
    if not _can_access(request.user):
        return JsonResponse({"error": "Access denied"}, status=403)
    try:
        ProductScan.objects.filter(id=scan_id).delete()
        return JsonResponse({"success": True})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
