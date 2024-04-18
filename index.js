const express = require('express');
const cors = require('cors');
const fs = require('fs');
const { spawn } = require('child_process')
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());

app.get('/tshirt', (req, res) => {

    const modelAI = spawn('python', ['Website_test.py'])

    modelAI.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    modelAI.on('close', (code) => {
        console.log(`python exited code ${code}`);
        
        const imagePath = path.join(__dirname, 'test.jpg');  // Path to your image file
        fs.readFile(imagePath, { encoding: 'base64' }, (err, data) => {
            if (err) {
            res.status(500).json({ error: 'Failed to load image' });
            return;
            }
            res.json({ image: `data:image/jpeg;base64,${data}` });
        });
    });
 });

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

