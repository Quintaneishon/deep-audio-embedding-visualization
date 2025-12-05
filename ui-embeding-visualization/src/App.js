import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { VisualizePage } from './pages/VisualizePage';

function App() {
  return (
    <Router>
      <div className='main'>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/visualize" element={<VisualizePage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
