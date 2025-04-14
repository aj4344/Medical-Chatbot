const express = require('express');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
const app = express();
const port = 3000;

const upload = multer({ dest: 'uploads/' });

app.post('/api/ask', upload.array('pdfs'), async (req, res) => {
  try {
    const question = req.body.question;
    const pdfFiles = req.files;
    console.log('Received question:', question);
    console.log('Received PDFs:', pdfFiles.map(f => f.originalname));

    if (!question || !pdfFiles) {
      console.log('Missing question or PDFs');
      return res.status(400).json({ answer: 'Missing question or PDFs!' });
    }

    const formData = new FormData();
    formData.append('question', question);
    pdfFiles.forEach(file => {
      formData.append('pdfs', fs.createReadStream(file.path), file.originalname);
    });

    console.log('Sending to Flask at http://localhost:5000/api/ask');
    const flaskResponse = await axios.post('http://localhost:5000/api/ask', formData, {
      headers: formData.getHeaders(),
    });

    pdfFiles.forEach(file => fs.unlinkSync(file.path));
    console.log('Flask response:', flaskResponse.data);
    res.json({ answer: flaskResponse.data.answer });
  } catch (error) {
    console.error('Error calling Flask:', error.message);
    res.status(500).json({ answer: `Error: ${error.message}` });
  }
});

app.listen(port, () => {
  console.log(`Node.js backend running on http://localhost:${port}`);
});
















// const express = require('express');
// const axios = require('axios');
// const multer = require('multer');
// const FormData = require('form-data');
// const fs = require('fs');
// const app = express();
// const port = 3000;

// const upload = multer({ dest: 'uploads/' });

// app.post('/api/process-pdfs', upload.array('pdfs'), async (req, res) => {
//   try {
//     const pdfFiles = req.files;
//     console.log('Received PDFs for processing:', pdfFiles.map(f => f.originalname));

//     if (!pdfFiles) {
//       console.log('No PDFs received');
//       return res.status(400).json({ message: 'No PDFs uploaded!' });
//     }

//     const formData = new FormData();
//     pdfFiles.forEach(file => {
//       formData.append('pdfs', fs.createReadStream(file.path), file.originalname);
//     });

//     console.log('Sending PDFs to Flask for processing');
//     const flaskResponse = await axios.post('http://localhost:5000/api/process-pdfs', formData, {
//       headers: formData.getHeaders(),
//     });
//     console.log('Flask processing response:', flaskResponse.data);

//     pdfFiles.forEach(file => fs.unlinkSync(file.path));
//     res.json({ message: flaskResponse.data.message });
//   } catch (error) {
//     console.error('Error processing PDFs:', error.message, error.response?.data);
//     res.status(500).json({ message: `Error: ${error.message}` });
//   }
// });

// app.post('/api/ask', async (req, res) => {
//   try {
//     const { question } = req.body;
//     console.log('Received question:', question);

//     if (!question) {
//       console.log('No question provided');
//       return res.status(400).json({ answer: 'No question provided!' });
//     }

//     console.log('Sending question to Flask');
//     const flaskResponse = await axios.post('http://localhost:5000/api/ask', { question }, {
//       headers: { 'Content-Type': 'application/json' },
//     });
//     console.log('Flask response:', flaskResponse.data);

//     res.json({ answer: flaskResponse.data.answer });
//   } catch (error) {
//     console.error('Error calling Flask:', error.message, error.response?.data);
//     res.status(500).json({ answer: `Error: ${error.message}` });
//   }
// });

// app.listen(port, () => {
//   console.log(`Node.js backend running on http://localhost:${port}`);
// });