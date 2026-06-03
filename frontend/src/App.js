import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import 'highlight.js/styles/tokyo-night-dark.css';

// Custom renderer for marked to open links in new tabs
const renderer = new marked.Renderer();
renderer.link = (href, title, text) => {
  return `<a href="${href}" title="${title || ''}" target="_blank" rel="noopener noreferrer">${text}</a>`;
};
marked.setOptions({ renderer });

function App() {
  const [messages, setMessages] = useState([
    { id: 'welcome', role: 'assistant', text: 'Hello! I am your AI assistant. How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(localStorage.getItem('session_id') || '');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async (e) => {
    if (e) e.preventDefault();
    const query = input.trim();
    if (!query || isTyping) return;

    setInput('');
    setIsTyping(true);

    // Append user message
    const userMsg = { id: Date.now().toString(), role: 'user', text: query };
    setMessages(prev => [...prev, userMsg]);

    // Append temporary bot message for streaming
    const botMsgId = (Date.now() + 1).toString();
    const newBotMsg = {
      id: botMsgId,
      role: 'assistant',
      text: '',
      mode: '',
      similarity: null,
      done: false
    };
    setMessages(prev => [...prev, newBotMsg]);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId || null,
          message: query
        })
      });

      setIsTyping(false);

      if (!response.ok) {
        throw new Error('Sorry, the server encountered an error processing your request.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let accumText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep last incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);

            if (data.session_id && !sessionId) {
              setSessionId(data.session_id);
              localStorage.setItem('session_id', data.session_id);
            }

            if (data.mode) {
              // eslint-disable-next-line no-loop-func
              setMessages(prev =>
                prev.map(m =>
                  m.id === botMsgId
                    ? { ...m, mode: data.mode, similarity: data.similarity }
                    : m
                )
              );
            }

            if (data.chunk) {
              accumText += data.chunk;
              // eslint-disable-next-line no-loop-func
              setMessages(prev =>
                prev.map(m =>
                  m.id === botMsgId
                    ? { ...m, text: accumText }
                    : m
                )
              );
            }

            if (data.done) {
              // eslint-disable-next-line no-loop-func
              setMessages(prev =>
                prev.map(m =>
                  m.id === botMsgId
                    ? { ...m, message_id: data.message_id, done: true }
                    : m
                )
              );
            }
          } catch (err) {
            console.error('Error parsing line:', err, line);
          }
        }
      }
    } catch (err) {
      setIsTyping(false);
      console.error(err);
      setMessages(prev =>
        prev.map(m =>
          m.id === botMsgId
            ? { ...m, text: err.message || 'Sorry, I encountered an error connecting to the server.', isError: true, done: true }
            : m
        )
      );
    }
  };

  const handleFeedback = async (messageId, rating, index) => {
    const msg = messages[index];
    if (msg.hasVoted) return;

    try {
      const res = await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message_id: messageId,
          rating: rating
        })
      });
      if (res.ok) {
        setMessages(prev =>
          prev.map((m, i) =>
            i === index ? { ...m, hasVoted: true, userRating: rating } : m
          )
        );
      }
    } catch (err) {
      console.error('Error submitting feedback:', err);
    }
  };

  const renderMessageText = (text) => {
    try {
      const html = marked.parse(text);
      return <div dangerouslySetInnerHTML={{ __html: html }} />;
    } catch (e) {
      return <div>{text}</div>;
    }
  };

  useEffect(() => {
    hljs.highlightAll();
  }, [messages]);

  return (
    <div className="app-container">
      <div className="chat-card">
        {/* Header */}
        <div className="chat-header">
          <div className="avatar">AI</div>
          <div className="header-info">
            <span className="title">AI Support Agent</span>
            <span className="status">Online</span>
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          {messages.map((msg, index) => (
            <div key={msg.id} className={`msg-wrapper ${msg.role === 'user' ? 'user-msg' : 'bot-msg'}`}>
              <div className={`msg-bubble ${msg.isError ? 'error-bubble' : ''}`}>
                {renderMessageText(msg.text)}
              </div>
              
              {msg.role === 'assistant' && msg.id !== 'welcome' && (
                <>
                  {msg.mode && (
                    <div className="meta-badge">
                      <span>mode: {msg.mode}</span>
                      {msg.similarity !== null && msg.similarity > 0 && (
                        <span>sim: {msg.similarity.toFixed(2)}</span>
                      )}
                    </div>
                  )}

                  {msg.done && msg.message_id && (
                    <div className="feedback-actions">
                      <button
                        className={`feedback-btn thumbs-up ${msg.hasVoted && msg.userRating === 1 ? 'active' : ''}`}
                        onClick={() => handleFeedback(msg.message_id, 1, index)}
                        disabled={msg.hasVoted}
                      >
                        👍 Helpful
                      </button>
                      <button
                        className={`feedback-btn thumbs-down ${msg.hasVoted && msg.userRating === -1 ? 'active' : ''}`}
                        onClick={() => handleFeedback(msg.message_id, -1, index)}
                        disabled={msg.hasVoted}
                      >
                        👎 Unhelpful
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="typing-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <form onSubmit={handleSend} className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isTyping}
          />
          <button type="submit" className="send-btn" disabled={isTyping || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
