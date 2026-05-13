from django.urls import path
from . import views

app_name = "product_scanner"

urlpatterns = [
    path("",                              views.scanner_page,     name="scanner"),
    path("api/scan/",                     views.api_scan,         name="api_scan"),
    path("api/scan/upload/",              views.api_scan_upload,  name="api_scan_upload"),
    path("api/history/",                  views.api_history,      name="api_history"),
    path("api/scan/<int:scan_id>/delete/",views.api_delete_scan,  name="api_delete"),
]
