function fmt(ts){ const d=new Date(ts); const p=n=>n<10?'0'+n:n; return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}` }

Page({
  data: { items: [] },
  onShow(){ this.load(); },
  load(){
    try{
      const list = wx.getStorageSync('learning_history') || [];
      const items = list.map(it => ({...it, time: fmt(it.ts)}));
      this.setData({ items });
    }catch(e){ this.setData({ items: []}); }
  },
  onClear(){
    try{ wx.setStorageSync('learning_history', []); }catch(e){}
    this.load();
  }
}); 