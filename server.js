// server.js — раздаёт фронтенд на порту 3000
// Запуск: node server.js

const http = require('http');
const fs   = require('fs');
const path = require('path');

const PORT = 3000;
const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript',
    '.css':  'text/css',
    '.json': 'application/json',
};

http.createServer((req, res) => {
    let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
    const ext    = path.extname(filePath);

    if (!fs.existsSync(filePath)) {
        filePath = path.join(__dirname, 'index.html');
    }

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end('Not found');
            return;
        }
        res.writeHead(200, { 'Content-Type': MIME[ext] || 'text/plain' });
        res.end(data);
    });
}).listen(PORT, () => {
    console.log(`\nФронтенд: http://localhost:${PORT}`);
    console.log(`\nНе забудьте запустить Python API:`);
    console.log(`  pip install fastapi uvicorn python-multipart laspy numpy torch`);
    console.log(`  python api.py\n`);
});
