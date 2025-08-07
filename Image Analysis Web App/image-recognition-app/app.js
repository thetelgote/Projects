const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const AWS = require('aws-sdk');

const app = express();
const port = 3000;

// Direct AWS Config (NOT recommended for production)
AWS.config.update({
  accessKeyId: 'YOUR_ACCESS_KEY_ID',         // Replace with your key
  secretAccessKey: 'YOUR_SECRET_ACCESS_KEY', // Replace with your secret
  region: 'ap-southeast-1'                   // Change to your region
});

const rekognition = new AWS.Rekognition();

// EJS Setup
app.set('view engine', 'ejs');

// Static folder for uploads
app.use('/uploads', express.static('uploads'));

// Multer Setup
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: function (req, file, cb) {
    const uniqueName = file.fieldname + '-' + Date.now() + path.extname(file.originalname);
    cb(null, uniqueName);
  }
});

const upload = multer({ storage: storage });

// Routes
app.get('/', (req, res) => {
  res.render('index', { labels: null, imagePath: null });
});

app.post('/upload', upload.single('image'), (req, res) => {
  const imagePath = req.file.path;
  const imageBytes = fs.readFileSync(imagePath);

  const params = {
    Image: {
      Bytes: imageBytes
    },
    MaxLabels: 10,
    MinConfidence: 70
  };

  rekognition.detectLabels(params, (err, data) => {
    if (err) {
      console.error('Rekognition Error:', err);
      res.send('Error processing image');
    } else {
      const labels = data.Labels.map(label => ({
        Name: label.Name,
        Confidence: label.Confidence.toFixed(2)
      }));
      res.render('index', { labels, imagePath: '/' + imagePath });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
