self.addEventListener('install', e => {
  e.waitUntil(
    caches.open('sentinel-cache-v1').then(cache => {
      return cache.addAll(['/', '/manifest.json']);
    })
  );
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(response => response || fetch(e.request))
  );
});

self.addEventListener('push', e => {
  const data = e.data.json();
  const options = {
    body: data.body,
    icon: 'https://img.icons8.com/fluency/192/000000/gold-bar.png',
    badge: 'https://img.icons8.com/fluency/192/000000/gold-bar.png',
    vibrate: [200, 100, 200],
    tag: 'sentinel-setup',
    renotify: true
  };
  e.waitUntil(self.registration.showNotification(data.title, options));
});
