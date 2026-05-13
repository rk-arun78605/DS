from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('reset-password/', views.reset_password_view, name='reset_password'),
    path('change-password/', views.change_password_view, name='change_password'),

    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Staff: Submit incident
    path('submit/', views.submit_incident, name='submit_incident'),
    path('submit/check-photo/', views.submit_check_photo, name='submit_check_photo'),
    path('my-incidents/', views.my_incidents, name='my_incidents'),

    # Edit draft incident (by first-item pk, since drafts have no incident_no)
    path('edit/<int:draft_pk>/', views.edit_incident, name='edit_incident'),

    # Supervisor: Approve
    path('approvals/', views.approvals_list, name='approvals_list'),
    path('approvals/<int:pk>/', views.approval_detail, name='approval_detail'),
    path('approvals/<int:pk>/upload-media/', views.upload_media, name='upload_media'),
    path('approvals/<int:pk>/media/<str:media_type>/', views.serve_media, name='serve_media'),
    path('item/<int:item_pk>/expiry-photo/', views.serve_item_expiry_photo, name='serve_item_expiry_photo'),
    path('approvals/<int:pk>/approve/', views.approve_incident, name='approve_incident'),
    path('approvals/<int:pk>/reject/', views.reject_incident, name='reject_incident'),

    # Analytics Dashboard
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    path('analytics/export/top-items/', views.analytics_export_top_items, name='analytics_export_top_items'),
    path('analytics/ai-chat/', views.analytics_ai_chat, name='analytics_ai_chat'),

    # Reports (supervisor/manager)
    path('reports/', views.reports, name='reports'),
    path('reports/daily/', views.daily_report, name='daily_report'),

    # Change 9: WIP master list
    path('wip/', views.wip_master_list, name='wip_master_list'),

    # Change 10: Geofencing check
    path('check-location/', views.check_location, name='check_location'),

    # AJAX
    path('api/item-search/', views.item_search_api, name='item_search_api'),
    path('api/item-detail/<str:item_code>/', views.item_detail_api, name='item_detail_api'),

    # User Management (requires separate MGMT auth)
    path('user-management/',                    views.um_user_list,   name='um_user_list'),
    path('user-management/login/',              views.um_login,       name='um_login'),
    path('user-management/logout/',             views.um_logout,      name='um_logout'),
    path('user-management/create/',             views.um_create_user, name='um_create_user'),
    path('user-management/edit/<str:emp_id>/',  views.um_edit_user,   name='um_edit_user'),
    path('user-management/toggle/<str:emp_id>/',views.um_toggle_user, name='um_toggle_user'),
    path('user-management/delete/<str:emp_id>/',views.um_delete_user, name='um_delete_user'),

    # MGMT master views (read-only)
    path('mgmt/', views.mgmt_dashboard, name='mgmt_dashboard'),
    path('mgmt/incident/<int:pk>/', views.mgmt_incident_detail, name='mgmt_incident_detail'),
    path('mgmt/incident/<int:pk>/media/<str:media_type>/', views.mgmt_serve_media, name='mgmt_serve_media'),

    # PWA
    path('manifest.json', views.pwa_manifest, name='pwa_manifest'),
    path('sw.js', views.service_worker, name='service_worker'),
    path('offline/', views.offline_view, name='offline'),
]
