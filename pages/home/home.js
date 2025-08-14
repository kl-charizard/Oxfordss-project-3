const request = require('../../utils/request.js');

Page({
  data: {
    suggestion: {
      word: '…',
      topic: '—',
      level: '—',
      hint: 'Fetching today\'s suggestion'
    }
  },
  onLoad() {
    // Create or reuse session id
    let sessionId = wx.getStorageSync('session_id');
    if (!sessionId) {
      sessionId = `${Date.now()}-${Math.floor(Math.random()*1e6)}`;
      wx.setStorageSync('session_id', sessionId);
    }
    this.fetchSuggestion(sessionId);
  },
  async fetchSuggestion(sessionId) {
    try {
      const res = await request.post('/chat', {
        session_id: sessionId,
        message: 'Give me one daily vocabulary suggestion: pick a random topic and level, output as: word | topic | level | one-sentence hint.'
      });

      // 1) Prefer structured payload from server
      if (res && Array.isArray(res.learned) && res.learned.length > 0) {
        const item = res.learned[0];
        this.setData({ suggestion: item });
        this._saveHistoryItem(item);
        return;
      }

      const text = (res && res.reply) ? String(res.reply) : '';
      const cleanText = text.replace(/<learned_json>[\s\S]*?<\/learned_json>/g, '').trim();

      // 2) Try to extract learned_json from reply text
      const m = text.match(/<learned_json>\s*([\s\S]*?)\s*<\/learned_json>/);
      if (m && m[1]) {
        try {
          const arr = JSON.parse(m[1]);
          if (Array.isArray(arr) && arr.length > 0) {
            const item = arr[0];
            this.setData({ suggestion: item });
            this._saveHistoryItem(item);
            return;
          }
        } catch (e) {
          // ignore and continue to pipe fallback
        }
      }

      // 3) Fallback: parse pipe-delimited "word | topic | level | hint"
      const parts = cleanText.split('|').map(s => s.trim());
      if (parts.length >= 4) {
        const item = { word: parts[0], topic: parts[1], level: parts[2], hint: parts.slice(3).join(' | ') };
        this.setData({ suggestion: item });
        this._saveHistoryItem(item);
        return;
      }

      // 4) Last resort
      this.setData({ suggestion: { word: 'Word', topic: 'General', level: 'All', hint: cleanText || 'Start learning now!' } });
      wx.showToast({ title: 'Daily suggestion parse failed', icon: 'none' });
    } catch (e) {
      console.error('fetchSuggestion error', e);
      this.setData({ suggestion: { word: 'Word', topic: 'General', level: 'All', hint: 'Start learning now!' } });
    }
  },
  _saveHistoryItem(item) {
    try {
      const list = wx.getStorageSync('learning_history') || [];
      list.unshift({ ...item, ts: Date.now() });
      wx.setStorageSync('learning_history', list.slice(0, 200));
    } catch (e) {}
  },
  goChat() {
    wx.navigateTo({ url: '/pages/chat/chat' });
  }
}); 