const http = require('http');
const { exec } = require('child_process');

const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });

    // Your C code
    const cCode = `
        #include <stdio.h>
        int main() {
            printf("Hello, World!");
            return 0;
        }
    `;

    // Compile and execute C code
    exec('gcc -o myCProgram -x c -', (err) => {
        if (err) {
            res.end(`<pre>Error: ${err.message}</pre>`);
            return;
        }

        exec('./myCProgram', (error, stdout, stderr) => {
            if (error) {
                res.end(`<pre>Error: ${error.message}</pre>`);
                return;
            }
            if (stderr) {
                res.end(`<pre>stderr: ${stderr}</pre>`);
                return;
            }
            res.end(`<pre>${stdout}</pre>`);
        }).stdin.write(cCode);
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
});
