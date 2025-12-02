import { useState } from 'react';
import { api } from '../api/client';

type Props = {
  onUploaded: () => void;
};

export function UploadPanel({ onUploaded }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState('');

  const handleFiles = async (files: FileList | null) => {
    if (!files || !files.length) return;
    const file = files[0];
    setStatus('Uploading...');
    try {
      const res = await api.uploadFile(file);
      setStatus('Upload complete');
      onUploaded();
    } catch (e: any) {
      setStatus(e?.message || 'Upload failed');
    }
  };

  return (
    <div className="upload-panel">
      <div
        className={`dropzone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
      >
        <div className="drop-icon">📄</div>
        <div className="drop-text">Drag research PDFs here, or <span className="highlight">browse</span>.</div>
        <input type="file" onChange={(e) => handleFiles(e.target.files)} />
      </div>
      <button className="primary-btn" onClick={() => document.querySelector<HTMLInputElement>('.dropzone input')?.click()}>
        + Upload Source
      </button>
      {status && <div className="status-text">{status}</div>}
    </div>
  );
}
