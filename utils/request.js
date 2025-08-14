const DEFAULTS = ['http://127.0.0.1:8000', 'http://localhost:8000']

function getStoredBaseUrl() {
  try {
    const v = wx.getStorageSync('BASE_URL')
    return v || DEFAULTS[0]
  } catch (e) {
    return DEFAULTS[0]
  }
}

function setBaseUrl(url) {
  try { wx.setStorageSync('BASE_URL', url) } catch (e) {}
}

function withFallback(requestFn) {
  return new Promise((resolve, reject) => {
    const first = getStoredBaseUrl()
    const alt = first.includes('127.0.0.1') ? DEFAULTS[1] : DEFAULTS[0]

    requestFn(first)
      .then(resolve)
      .catch(() => {
        // try fallback once
        setBaseUrl(alt)
        requestFn(alt).then(resolve).catch(reject)
      })
  })
}

function _wxRequest(method, path, data) {
  return withFallback((base) => new Promise((resolve, reject) => {
    wx.request({
      url: `${base}${path}`,
      method,
      data,
      header: { 'content-type': 'application/json' },
      success(res) {
        const { statusCode, data } = res
        if (statusCode >= 200 && statusCode < 300) {
          resolve({ data, base })
        } else {
          reject(new Error(`HTTP ${statusCode}: ${typeof data === 'string' ? data : JSON.stringify(data)}`))
        }
      },
      fail(err) { reject(err) }
    })
  }))
}

function post(path, data) {
  return _wxRequest('POST', path, data).then((res) => {
    // persist working base
    if (res && res.base) setBaseUrl(res.base)
    return res.data
  })
}

function get(path) {
  return _wxRequest('GET', path, undefined).then((res) => {
    if (res && res.base) setBaseUrl(res.base)
    return res.data
  })
}

function ping() { return get('/health') }

module.exports = { post, get, ping, getStoredBaseUrl, setBaseUrl }; 