import { useState } from 'react';

type Props = {
  apiBase: string;
};

export function UploadPanel({ apiBase }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');

  const onUpload = async () => {
    if (!file) return;
    setStatus('Uploading...');
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${apiBase}/documents/upload`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      setStatus('Upload failed');
      return;
    }
    const data = await res.json();
    setStatus(`Uploaded doc #${data.document_id}, chunks: ${data.chunks}`);
  };

  return (
    <div className="card upload">
      <h3>Upload PDF/Text</h3>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={onUpload} disabled={!file}>Upload</button>
      <div className="status">{status}</div>
    </div>
  );
}
