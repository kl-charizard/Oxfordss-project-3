const request = require('../../utils/request.js');

Page({
  data: {
    messages: [],
    inputText: '',
    sending: false,
    clearing: false,
    lastMessageId: '',
    sessionId: '',
    sessionShort: ''
  },
  onLoad() {
    try {
      let sessionId = wx.getStorageSync('session_id');
      if (!sessionId) {
        sessionId = `${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
        wx.setStorageSync('session_id', sessionId);
      }

      this.setData({ sessionId, sessionShort: this._short(sessionId) });

      // Optional connectivity toast
      request.ping()
        .then(() => {
          const base = request.getStoredBaseUrl();
          wx.showToast({ title: `Connected: ${base.replace('http://','')}`, icon: 'none', duration: 1500 });
        })
        .catch(() => {
          const base = request.getStoredBaseUrl();
          wx.showToast({ title: `Server unreachable at ${base}`, icon: 'none', duration: 2000 });
        });
    } catch (e) {
      console.error('onLoad error', e);
    }
  },
  onInput(e) {
    this.setData({ inputText: e.detail.value });
  },
  onSend() {
    const text = (this.data.inputText || '').trim();
    if (!text) return;

    this.addUser(text);
    this.setData({ inputText: '', sending: true });

    request.post('/chat', {
      session_id: this.data.sessionId,
      message: text,
      mode: 'chat'
    })
    .then((res) => {
      const reply = res && res.reply ? String(res.reply) : 'No reply';
      // Hide the learned_json block from chat display, but still use it for history
      const cleaned = reply.replace(/<learned_json>[\s\S]*?<\/learned_json>/g, '').trim();
      this.addBot(cleaned || '');
      if (res && Array.isArray(res.learned) && res.learned.length > 0) {
        this._saveHistory(res.learned);
      }
    })
    .catch((err) => {
      console.error('chat error', err);
      const base = request.getStoredBaseUrl();
      wx.showToast({ title: `Request failed (${base})`, icon: 'none' });
      this.addBot('Sorry, I had trouble reaching the server.');
    })
    .finally(() => {
      this.setData({ sending: false });
    });
  },
  onClearMemory() {
    if (!this.data.sessionId) return;
    this.setData({ clearing: true });
    request.post('/reset', { session_id: this.data.sessionId })
      .then(() => {
        this.setData({ messages: [] });
      })
      .catch((err) => {
        console.error('reset error', err);
        const base = request.getStoredBaseUrl();
        wx.showToast({ title: `Reset failed (${base})`, icon: 'none' });
      })
      .finally(() => this.setData({ clearing: false }));
  },
  onNewSession() {
    const newId = this._genSessionId();
    wx.setStorageSync('session_id', newId);
    this.setData({ sessionId: newId, sessionShort: this._short(newId), messages: [] });
  },
  _saveHistory(items) {
    try {
      const list = wx.getStorageSync('learning_history') || [];
      const withTs = items.map(it => ({ ...it, ts: Date.now() }));
      wx.setStorageSync('learning_history', withTs.concat(list).slice(0, 200));
    } catch (e) {}
  },
  addUser(text) {
    const id = `m${Date.now()}u`;
    const msg = { id, role: 'user', text };
    const messages = this.data.messages.concat(msg);
    this.setData({ messages, lastMessageId: `msg-${id}` });
  },
  addBot(text) {
    const id = `m${Date.now()}b`;
    const msg = { id, role: 'bot', text };
    const messages = this.data.messages.concat(msg);
    this.setData({ messages, lastMessageId: `msg-${id}` });
  },
  _genSessionId() {
    return `${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
  },
  _short(id) {
    if (!id) return '';
    const parts = String(id).split('-');
    return parts[1] ? parts[1] : id.slice(-6);
  }
}); 