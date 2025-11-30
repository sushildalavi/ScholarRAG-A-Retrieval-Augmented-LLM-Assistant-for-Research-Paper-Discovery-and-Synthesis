import React, { useEffect, useState } from "react";

export function AnswerBox({ text, loading }) {
  const [display, setDisplay] = useState("");

  useEffect(() => {
    if (!text) {
      setDisplay("");
      return;
    }
    let idx = 0;
    const words = text.split(" ");
    const id = setInterval(() => {
      idx += 1;
      setDisplay(words.slice(0, idx).join(" "));
      if (idx >= words.length) clearInterval(id);
    }, 25);
    return () => clearInterval(id);
  }, [text]);

  return (
    <div className="card" style={{ padding: 16, marginTop: 10, minHeight: 90 }}>
      {loading ? <div className="skeleton" style={{ height: 24, width: "70%", borderRadius: 8 }} /> : <p style={{ margin: 0, lineHeight: 1.6 }}>{display}</p>}
    </div>
  );
}
