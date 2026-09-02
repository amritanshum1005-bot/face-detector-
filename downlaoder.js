import fs from 'fs';
import path from 'path';
import https from 'https';

const modelsDir = path.join(process.cwd(), 'public', 'models');

// Ensure models directory exists
if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
}

const BASE_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights/';

const files = [
    'tiny_face_detector_model-weights_manifest.json',
    'tiny_face_detector_model-shard1',
    'face_expression_model-weights_manifest.json',
    'face_expression_model-shard1',
    'age_gender_model-weights_manifest.json',
    'age_gender_model-shard1',
    'face_landmark_68_model-weights_manifest.json',
    'face_landmark_68_model-shard1'
];

function downloadFile(url, destPath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath);
        https.get(url, (response) => {
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download ${url}: HTTP Status ${response.statusCode}`));
                return;
            }
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                console.log(`Successfully downloaded: ${path.basename(destPath)}`);
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(destPath, () => { }); // delete partial file
            reject(err);
        });
    });
}

async function run() {
    console.log('Downloading face-api.js weights to public/models/...');
    for (const filename of files) {
        const fileUrl = `${BASE_URL}${filename}`;
        const destPath = path.join(modelsDir, filename);
        try {
            await downloadFile(fileUrl, destPath);
        } catch (err) {
            console.error(`Error downloading ${filename}:`, err.message);
            process.exit(1);
        }
    }
    console.log('All model weights downloaded successfully!');
}

run();
