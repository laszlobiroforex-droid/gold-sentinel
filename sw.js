// sw.js - Service Worker for Background Notifications
self.addEventListener('push', function(event) {
    const data = event.data.json();
    const options = {
        body: data.body,
        icon: 'https://cdn-icons-png.flaticon.com/512/2983/2983804.png',
        badge: 'https://cdn-icons-png.flaticon.com/512/2983/2983804.png',
        vibrate: [200, 100, 200]
    };
    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

self.addEventListener('notificationclick', function(event) {
    event.notification.close();
    event.waitUntil(
        clients.openWindow('https://your-app-url.streamlit.app')
    );
});
