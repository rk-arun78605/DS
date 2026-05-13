# Mobile App Deployment Guide - Melcom KPI Dashboard

## Overview
This guide shows 3 methods to make your Streamlit dashboard accessible as a mobile app.

---

## Method 1: Streamlit Cloud (Easiest - FREE)

### ✅ Advantages
- No server management
- Automatic HTTPS
- Free tier available
- Updates automatically on git push
- Perfect for internal use

### 📋 Steps

1. **Prepare your repository:**
```bash
# Create requirements.txt
pip freeze > requirements.txt

# Create .streamlit/config.toml for mobile optimization
mkdir .streamlit
```

2. **Create `.streamlit/config.toml`:**
```toml
[server]
headless = true
port = 8503

[theme]
primaryColor = "#8B0000"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[browser]
gatherUsageStats = false
```

3. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Deploy KPI Dashboard"
git push origin main
```

4. **Deploy on Streamlit Cloud:**
- Visit https://share.streamlit.io/
- Sign in with GitHub
- Click "New app"
- Select your repository
- Main file: `kpi_app/kpi_dashboard.py`
- Click "Deploy"

5. **Result:**
- You get: `https://your-app-name.streamlit.app`
- Works on any device with browser
- Automatic SSL certificate

---

## Method 2: Progressive Web App (PWA) - Best Mobile Experience

### ✅ Advantages
- **Add to Home Screen** - looks like native app
- Offline capability (with service worker)
- Full screen mode (no browser UI)
- Push notifications (optional)
- Works on iOS and Android

### 📋 Implementation

**Step 1: Create `manifest.json` in your app folder:**
```json
{
  "name": "Melcom KPI Dashboard",
  "short_name": "KPI",
  "description": "Melcom Retail Analytics Dashboard",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#FFFFFF",
  "theme_color": "#8B0000",
  "orientation": "portrait-primary",
  "icons": [
    {
      "src": "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ]
}
```

**Step 2: Add PWA meta tags to your dashboard:**

Add this to your `kpi_dashboard.py` after the custom CSS:

```python
# Add PWA support
st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="KPI Dashboard">
<meta name="theme-color" content="#8B0000">
<link rel="apple-touch-icon" href="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg">
""", unsafe_allow_html=True)
```

**Step 3: Serve manifest.json:**

Create `serve_manifest.py`:
```python
from flask import Flask, send_from_directory
import subprocess
import threading

app = Flask(__name__)

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('.', 'manifest.json')

def run_streamlit():
    subprocess.run(['streamlit', 'run', 'kpi_dashboard.py', '--server.port', '8503'])

if __name__ == '__main__':
    # Run Streamlit in background thread
    threading.Thread(target=run_streamlit, daemon=True).start()
    # Serve manifest on port 8000
    app.run(host='0.0.0.0', port=8000)
```

**Step 4: Mobile-optimized CSS (already added to your dashboard):**
```python
# Mobile responsive viewport (add to custom CSS)
st.markdown("""
<style>
@media (max-width: 768px) {
    .block-container {
        padding: 1rem !important;
    }
    
    .kpi-card {
        font-size: 0.85rem !important;
    }
    
    /* Stack columns on mobile */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    
    /* Smaller text on mobile */
    .kpi-value {
        font-size: 1.5rem !important;
    }
    
    /* Hide sidebar by default on mobile */
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 250px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 0px !important;
        margin-left: -250px;
    }
}

/* Landscape mode optimization */
@media (orientation: landscape) and (max-height: 500px) {
    .block-container {
        padding-top: 0.5rem !important;
    }
}

/* Touch-friendly buttons */
button {
    min-height: 44px !important;
    min-width: 44px !important;
}
</style>
""", unsafe_allow_html=True)
```

**Step 5: Test PWA:**
1. Deploy to server (see Method 3 below)
2. Open on mobile browser: `https://your-server-ip:8503`
3. Chrome Android: Three dots → "Add to Home Screen"
4. Safari iOS: Share button → "Add to Home Screen"
5. App icon appears on home screen like native app

---

## Method 3: Docker Deployment (Production-Ready)

### ✅ Advantages
- Full control
- Can deploy anywhere (AWS, Azure, Digital Ocean)
- Scalable
- Isolated environment

### 📋 Steps

**Step 1: Create `Dockerfile` in your project root:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8503

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "kpi_app/kpi_dashboard.py", "--server.port=8503", "--server.address=0.0.0.0", "--server.headless=true"]
```

**Step 2: Create `docker-compose.yml`:**
```yaml
version: '3.8'

services:
  kpi-dashboard:
    build: .
    ports:
      - "8503:8503"
    environment:
      - DB_HOST=host.docker.internal  # Windows/Mac
      - DB_PORT=3307
      - DB_USER=postgres
      - DB_PASSWORD=hello
    restart: unless-stopped
    networks:
      - melcom-network

  # Optional: Add Nginx reverse proxy for HTTPS
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - kpi-dashboard
    networks:
      - melcom-network

networks:
  melcom-network:
    driver: bridge
```

**Step 3: Build and run:**
```bash
# Build image
docker build -t melcom-kpi .

# Run container
docker-compose up -d

# View logs
docker-compose logs -f kpi-dashboard

# Access: http://localhost:8503
```

**Step 4: Deploy to cloud (example - AWS EC2):**
```bash
# 1. SSH to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Clone your repository
git clone your-repo-url
cd your-repo

# 4. Run docker-compose
sudo docker-compose up -d

# 5. Access via EC2 public IP
# http://your-ec2-ip:8503
```

---

## Method 4: Windows Server Deployment (Your Network)

### ✅ Best for: Internal company use, already have Windows server

**Step 1: Install Python on Windows Server**
```powershell
# Download Python 3.10+ installer
# Install with "Add to PATH" checked
python --version  # Verify
```

**Step 2: Create Windows Service**

Create `run_kpi_service.py`:
```python
import subprocess
import time
import sys

def run_dashboard():
    while True:
        try:
            print("Starting KPI Dashboard...")
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run',
                'kpi_app/kpi_dashboard.py',
                '--server.port=8503',
                '--server.address=0.0.0.0',
                '--server.headless=true'
            ])
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)  # Restart after 30 seconds

if __name__ == '__main__':
    run_dashboard()
```

**Step 3: Create scheduled task (auto-start on boot):**
```powershell
# Create task that runs on startup
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "D:\Dashboard Code\NO_WH\DS\run_kpi_service.py" -WorkingDirectory "D:\Dashboard Code\NO_WH\DS"

$trigger = New-ScheduledTaskTrigger -AtStartup

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "MelcomKPIDashboard" -Action $action -Trigger $trigger -Principal $principal -Description "Auto-start Melcom KPI Dashboard"

# Start immediately
Start-ScheduledTask -TaskName "MelcomKPIDashboard"
```

**Step 4: Configure firewall:**
```powershell
# Allow port 8503 through Windows Firewall
New-NetFirewallRule -DisplayName "Melcom KPI Dashboard" -Direction Inbound -LocalPort 8503 -Protocol TCP -Action Allow
```

**Step 5: Access from mobile:**
- Same WiFi: `http://server-ip:8503`
- External: Setup port forwarding on router (port 8503 → server IP)
- Secure: Use VPN or setup HTTPS with Let's Encrypt

---

## Recommended Approach for Melcom

### For Internal Use (Employees on company network):
**→ Use Method 4 (Windows Server) + Method 2 (PWA)**

1. Deploy on existing Windows server (already have PostgreSQL there)
2. Add PWA manifest for "Add to Home Screen" capability
3. Employees access via company WiFi
4. No monthly costs
5. Full data control (stays in Ghana)

### For External/Public Access:
**→ Use Method 1 (Streamlit Cloud) or Method 3 (Docker on AWS)**

1. Lower cost: Streamlit Cloud (free tier)
2. More control: AWS EC2 + Docker ($5-20/month)
3. Proper HTTPS/SSL
4. No firewall/network issues

---

## Mobile Optimization Checklist

✅ Already implemented in your dashboard:
- [x] Responsive CSS media queries
- [x] Touch-friendly button sizes (min 44px)
- [x] Mobile-optimized columns (stack on small screens)
- [x] Reduced padding on mobile
- [x] Hide sidebar collapse on mobile
- [x] Fast loading (cache + optimized queries)

⚠️ Add these enhancements:
- [ ] Add manifest.json for PWA
- [ ] Add PWA meta tags
- [ ] Test on actual mobile devices (iOS + Android)
- [ ] Add offline capability (service worker)
- [ ] Add pull-to-refresh
- [ ] Add haptic feedback on button taps
- [ ] Optimize images for mobile (already using Melcom logo URL)

---

## Testing Mobile Experience

### Desktop Browser (Chrome DevTools):
```
1. Press F12
2. Click device toolbar icon (Ctrl+Shift+M)
3. Select device: iPhone 12 Pro, Samsung Galaxy S21, etc.
4. Test portrait + landscape
5. Test touch events
6. Test network throttling (3G/4G)
```

### Real Device Testing:
```
1. Connect mobile to same WiFi as server
2. Find server IP: ipconfig (Windows) or ifconfig (Linux)
3. Open mobile browser: http://192.168.1.x:8503
4. Test all features
5. Check responsiveness
6. Test "Add to Home Screen"
```

---

## Security Considerations

### For Production Deployment:

1. **Enable HTTPS:**
```python
# Use Nginx reverse proxy with Let's Encrypt SSL
# OR use Streamlit Cloud (automatic HTTPS)
```

2. **Add authentication (already have in your dashboard):**
```python
# Keep your current authentication in session_state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
```

3. **Database security:**
- Don't hardcode passwords in code
- Use environment variables:
```python
import os
DB_PASSWORD = os.getenv('DB_PASSWORD', 'hello')
```

4. **Rate limiting:**
```python
# Add to nginx.conf if using reverse proxy
limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
```

---

## Cost Comparison

| Method | Setup Time | Monthly Cost | Maintenance | Best For |
|--------|-----------|--------------|-------------|----------|
| Streamlit Cloud | 10 min | FREE ($0) | None | Quick start, demos |
| Windows Server | 1 hour | $0 (use existing) | Low | Internal use |
| AWS EC2 | 2 hours | $5-50 | Medium | Production, external |
| Docker + AWS | 3 hours | $10-100 | Medium | Scalable, professional |

---

## Next Steps

1. **Test current setup on mobile:**
   - Open `http://your-server-ip:8503` on phone
   - Check responsiveness
   - Test all features

2. **Add PWA capability** (30 minutes):
   - Create manifest.json
   - Add meta tags
   - Test "Add to Home Screen"

3. **Choose deployment method:**
   - Internal only → Windows Server + PWA
   - External access → Streamlit Cloud or AWS

4. **Deploy:**
   - Follow steps for chosen method
   - Test on multiple devices
   - Share with users

---

## Support Resources

- Streamlit Deployment: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- PWA Guide: https://web.dev/progressive-web-apps/
- Docker Streamlit: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
- Mobile Testing: https://developer.chrome.com/docs/devtools/device-mode/

---

## Questions?

Common issues and solutions:

**Q: Dashboard looks good on desktop but broken on mobile?**
A: Check media queries, test with Chrome DevTools device mode first

**Q: Can't access from phone on same WiFi?**
A: Check Windows Firewall allows port 8503, verify server IP with `ipconfig`

**Q: "Add to Home Screen" not showing?**
A: Need HTTPS (use Streamlit Cloud or Nginx with Let's Encrypt)

**Q: Dashboard too slow on mobile 3G?**
A: Already optimized with caching; consider reducing data displayed per page

**Q: Want custom domain (kpi.melcom.com)?**
A: Setup DNS A record → server IP, use Nginx reverse proxy with SSL

---

Generated: 2025-01-20
For: Melcom KPI Dashboard
Version: 1.0
