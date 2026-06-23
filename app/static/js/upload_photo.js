// ── Elements ──────────────────────────────────────────────────────────────────
const fileInput = document.getElementById('fileInput')
const fileLabel = document.getElementById('fileLabel')
const selectBtn = document.getElementById('selectBtn')
const uploadSubmit = document.getElementById('uploadSubmit')
const resultSection = document.getElementById('resultSection')

const cameraVideo = document.getElementById('cameraVideo')
const canvas = document.getElementById('canvas')
const previewContainer = document.getElementById('previewContainer')
const cameraPlaceholder = document.getElementById('cameraPlaceholder')

const startCameraBtn = document.getElementById('startCameraBtn')
const captureBtn = document.getElementById('captureBtn')
const stopCameraBtn = document.getElementById('stopCameraBtn')
const retakeBtn = document.getElementById('retakeBtn')
const analyzeBtn = document.getElementById('analyzeBtn')

let stream = null
let capturedB64 = null

// ── Helpers ───────────────────────────────────────────────────────────────────
const show = (...els) => els.forEach(el => el?.classList.remove('hidden'))
const hide = (...els) => els.forEach(el => el?.classList.add('hidden'))

function showVideo() {
  cameraPlaceholder.style.display = 'none'
  cameraVideo.style.display = 'block'
  previewContainer.innerHTML = ''
}
function showPlaceholder() {
  cameraVideo.style.display = 'none'
  cameraPlaceholder.style.display = 'flex'
  previewContainer.innerHTML = ''
}
function showPreview(src) {
  cameraVideo.style.display = 'none'
  cameraPlaceholder.style.display = 'none'
  previewContainer.innerHTML = `
    <img src="${src}" style="width:100%;max-width:560px;border-radius:8px;
      border:1px solid rgba(0,240,192,0.2);box-shadow:0 0 24px rgba(0,240,192,0.15);" />`
}
function stopStream() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null }
}

// ── Result renderer ───────────────────────────────────────────────────────────
function renderResult(data) {
  const items = (data.detections || []).map(d => `
    <div class="detection-item ${d.label === 'Mask' ? 'mask' : 'no-mask'}">
      <strong>${d.label === 'Mask' ? 'Mask ✓' : 'No Mask ✗'}</strong>
      Face : ${d.face_num}<br/>Confidence: ${d.confidence}
    </div>`).join('')

  resultSection.innerHTML = `
    <div class="summary">${data.detection}</div>
    <img src="${data.image_b64}" class="result-image" alt="Detection result" />
    <div class="detection-list">${items}</div>`

  resultSection.style.display = 'flex'
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function setLoading(btn, isLoading, defaultText, loadingText) {
  btn.disabled = isLoading
  btn.textContent = isLoading ? loadingText : defaultText
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const leavingCapture = document.getElementById('capture-tab')?.classList.contains('active')
    if (leavingCapture && btn.dataset.tab !== 'capture') {
      stopStream(); resetCameraUI()
    }
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'))
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'))
    btn.classList.add('active')
    document.getElementById(btn.dataset.tab + '-tab').classList.add('active')
  })
})

// ── Upload tab ────────────────────────────────────────────────────────────────
selectBtn?.addEventListener('click', () => fileInput.click())

fileInput?.addEventListener('change', e => {
  fileLabel.textContent = e.target.files[0]?.name || 'No file chosen'
})

uploadSubmit?.addEventListener('click', async () => {
  const file = fileInput?.files[0]
  if (!file) { alert('Please select an image first.'); return }

  setLoading(uploadSubmit, true, '🚀 Start Detection', '⏳ Detecting...')
  resultSection.style.display = 'none'

  try {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch('/upload_photo', { method: 'POST', body: fd })
    const data = await res.json()

    if (data.success) {
      renderResult(data)
    } else {
      alert('❌ ' + (data.error || 'Detection failed'))
    }
  } catch (err) {
    console.error(err)
    alert('❌ Network error. Please try again.')
  } finally {
    setLoading(uploadSubmit, false, '🚀 Start Detection', '')
  }
})

// ── Camera flow ───────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    // Check if navigator.mediaDevices is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('❌ Camera access is not supported in this browser or context.\n\nMake sure:\n1. Using HTTPS (not HTTP)\n2. Browser supports Camera API\n3. Running on localhost or secure domain')
      return
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    })
    cameraVideo.srcObject = stream
    await new Promise(resolve => { cameraVideo.onloadedmetadata = () => { cameraVideo.play(); resolve() } })
    showVideo()
    hide(startCameraBtn, retakeBtn, analyzeBtn)
    show(captureBtn, stopCameraBtn)
    captureBtn.disabled = false
  } catch (err) {
    let msg = '❌ Camera error: ' + err.message
    if (err.name === 'NotAllowedError') msg = '❌ Camera permission denied. Please allow access and retry.'
    else if (err.name === 'NotFoundError') msg = '❌ No camera found. Connect a webcam and retry.'
    alert(msg)
  }
}

function resetCameraUI() {
  capturedB64 = null
  showPlaceholder()
  show(startCameraBtn)
  hide(captureBtn, stopCameraBtn, retakeBtn, analyzeBtn)
  captureBtn.disabled = true
}

startCameraBtn?.addEventListener('click', startCamera)

stopCameraBtn?.addEventListener('click', () => { stopStream(); resetCameraUI() })

captureBtn?.addEventListener('click', () => {
  if (!stream) return

  canvas.width = cameraVideo.videoWidth || 640
  canvas.height = cameraVideo.videoHeight || 480

  const ctx = canvas.getContext('2d')

  // ── Mirror fix ────────────────────────────────────────────────────────────
  // The browser shows the webcam preview as a mirror (selfie-style).
  // canvas.toDataURL() captures RAW unmirrored bytes by default, which means
  // the captured image is flipped compared to what the user sees.
  // We manually flip the canvas so the captured image matches the preview,
  // AND matches the orientation the ML model was trained on.
  ctx.save()
  ctx.translate(canvas.width, 0)   // move origin to right edge
  ctx.scale(-1, 1)                  // flip horizontally
  ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height)
  ctx.restore()
  // ─────────────────────────────────────────────────────────────────────────

  capturedB64 = canvas.toDataURL('image/jpeg', 0.92)
  stopStream()
  showPreview(capturedB64)
  hide(captureBtn, stopCameraBtn, startCameraBtn)
  show(retakeBtn, analyzeBtn)
  analyzeBtn.disabled = false
  analyzeBtn.textContent = '✅ Analyze Mask'
})

retakeBtn?.addEventListener('click', () => { capturedB64 = null; hide(retakeBtn, analyzeBtn); startCamera() })

analyzeBtn?.addEventListener('click', async () => {
  if (!capturedB64) return
  setLoading(analyzeBtn, true, '✅ Analyze Mask', '⏳ Analyzing...')
  resultSection.style.display = 'none'

  try {
    const res = await fetch('/upload_photo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: capturedB64 }),
    })
    const data = await res.json()
    if (data.success) {
      renderResult(data)
    } else {
      alert('❌ ' + (data.error || 'Detection failed'))
      setLoading(analyzeBtn, false, '✅ Analyze Mask', '')
    }
  } catch (err) {
    console.error(err)
    alert('❌ Network error.')
    setLoading(analyzeBtn, false, '✅ Analyze Mask', '')
  }
})

// ── Cleanup ───────────────────────────────────────────────────────────────────
window.addEventListener('beforeunload', stopStream)
document.addEventListener('visibilitychange', () => { if (document.hidden) stopStream() })