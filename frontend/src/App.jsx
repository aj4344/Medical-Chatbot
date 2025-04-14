import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [pdfs, setPdfs] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!pdfs.length) {
      setAnswer('Please upload at least one PDF first!');
      return;
    }
    setLoading(true);
    setAnswer('');
    try {
      const formData = new FormData();
      formData.append('question', question);
      pdfs.forEach((pdf) => formData.append('pdfs', pdf));
      const response = await axios.post('/api/ask', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setAnswer(response.data.answer);
    } catch (error) {
      setAnswer('Error: Couldn’t get an answer. Try again!');
    } finally {
      setLoading(false);
      setQuestion('');
    }
  };

  const handleFileUpload = (e) => {
    setPdfs([...e.target.files]);
  };

  return (
    <div className="app-container">
      <h1 className="app-title">💊 MediBot</h1>
      <p className="app-subtitle">Upload medical PDFs and ask questions!</p>

      <div className="upload-section">
        <label className="upload-label">Upload PDFs</label>
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileUpload}
          className="upload-input"
        />
      </div>

      <form onSubmit={handleSubmit} className="form">
        <label className="question-label">Your Question</label>
        <div className="input-group">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask your medical question..."
            disabled={loading}
            className="question-input"
          />
          <button type="submit" disabled={loading} className="ask-button">
            {loading ? (
              <span className="spinner"></span>
            ) : (
              'Ask MediBot'
            )}
          </button>
        </div>
      </form>

      {answer && (
        <div className="answer-box">
          <h3 className="answer-title">🤖 MediBot:</h3>
          <p className="answer-text">{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;