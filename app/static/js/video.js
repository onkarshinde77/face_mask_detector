/* video.js — AJAX upload, real % progress polling, in-browser video player */

document.addEventListener('DOMContentLoaded', () => {

  // ── Elements ──────────────────────────────────────────────────────────────
  const fileInput = document.getElementById('fileInput')
  const fileLabel = document.getElementById('fileLabel')
  const dropLabel = document.getElementById('dropLabel')
  const submitBtn = document.getElementById('submitBtn')

  const uploadContainer = document.getElementById('uploadContainer')
  const loadingContainer = document.getElementById('loadingContainer')
  const successContainer = document.getElementById('successContainer')
  const errorContainer = document.getElementById('errorContainer')
  const errorMessage = document.getElementById('errorMessage')

  const progressFill = document.getElementById('progressFill')
  const progressPct = document.getElementById('progressPct')
  const videoPlayer = document.getElementById('videoPlayer')
  const downloadBtn = document.getElementById('downloadBtn')

  let pollTimer = null

  // ── Helpers ───────────────────────────────────────────────────────────────
  function showPanel(name) {
    [uploadContainer, loadingContainer, successContainer, errorContainer]
      .forEach(el => el?.classList.remove('active'))
    const map = {
      upload: uploadContainer,
      loading: loadingContainer,
      success: successContainer,
      error: errorContainer,
    }
    map[name]?.classList.add('active')
  }

  function setProgress(pct) {
    const p = Math.min(Math.max(pct, 0), 100)
    progressFill.style.width = p + '%'
    progressPct.textContent = p + '%'
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
  }

  // ── File label update ─────────────────────────────────────────────────────
  fileInput?.addEventListener('change', e => {
    const name = e.target.files[0]?.name || 'Choose Video File'
    fileLabel.textContent = name
    dropLabel?.classList.toggle('has-file', !!e.target.files[0])
  })

  // ── Submit ────────────────────────────────────────────────────────────────
  submitBtn?.addEventListener('click', async () => {
    const file = fileInput?.files[0]
    if (!file) { alert('Please select a video file first.'); return }

    setProgress(0)
    showPanel('loading')
    submitBtn.disabled = true

    try {
      const fd = new FormData()
      fd.append('file', file)

      const res = await fetch('/upload_video', { method: 'POST', body: fd })
      console.log("Video upload response status:", res.status)

      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(`Server error ${res.status}: ${errorText}`)
      }

      const data = await res.clone().json()
      console.log("Json data:", data)

      if (data.success && data.video_id) {
        console.log("Starting polling...")
        startPolling(data.video_id)
        console.log("Polling started.")
      } else {
        showError(data.error || 'Upload failed.')
      }
    } catch (err) {
      showError('Network error during upload: ' + err.message)
    } finally {
      submitBtn.disabled = false
    }
  }, { once: true })

  // ── Poll /video_status/<id> every 1 second ────────────────────────────────
  function startPolling(videoId) {
    stopPolling()
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/video_status/${videoId}`)
        const data = await res.json()

        // Update progress bar
        if (typeof data.progress === 'number') {
          setProgress(data.progress)
        }

        if (data.status === 'done') {
          stopPolling()
          setProgress(100)

          // Small delay so user sees 100% before success panel
          setTimeout(() => {
            // Set video player src for in-browser playback
            videoPlayer.src = data.stream_url
            videoPlayer.load()
            // Set download link
            downloadBtn.href = data.download_url
            showPanel('success')
          }, 400)
        }
        else if (data.status === 'error') {
          stopPolling()
          showError(data.error || 'Processing failed.')
        }
      } catch (err) {
        stopPolling()
        showError('Connection error while checking status.')
      }
    }, 1000)
  }

  function showError(msg) {
    errorMessage.textContent = '❌ ' + msg
    showPanel('error')
    submitBtn.disabled = false
  }

})
