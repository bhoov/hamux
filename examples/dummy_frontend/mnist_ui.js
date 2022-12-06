// Minimal javascript UI that to draw digits & output results.
//
// Usage:
//
// const ui = mnist_ui(document.querySelector('#container'));
// ui.onUpdate(img => {
//   const {values, indices} = computeProbabilities(img);
//   ui.showPreds(values, indices);
// });

const mnist_ui = (target) => {

    // Image size. Must match model input layer size.
    let sz = 28
    // Bigger numbers give thicker strokes...
    let upscaleFactor = 8, halfPenSize = 1
    // Labels to name top predictions.
    const labels = ['zero ', 'one  ', 'two  ', 'three', 'four ', 'five ', 'six  ', 'seven', 'eight', 'nine ']
  
    const canvas = document.createElement('canvas')
    target.append(canvas)
    canvas.width = canvas.height = sz * upscaleFactor
    const clear = document.createElement('button')
    clear.innerText = 'clear'
    target.append(clear)
    const output = document.createElement('pre')
    target.append(output)
  
    let ctx = canvas.getContext('2d')
    let img = new Uint8Array(sz * sz)
    let dragging = false
    let timeout
  
    const getPos = e => {
      let x = e.offsetX, y = e.offsetY
      if (e.touches) {
        const rect = canvas.getBoundingClientRect()
        x = e.touches[0].clientX - rect.left
        y = e.touches[0].clientY - rect.left
      }
      return {
        x: Math.floor((x - 2*halfPenSize*upscaleFactor/2)/upscaleFactor),
        y: Math.floor((y - 2*halfPenSize*upscaleFactor/2)/upscaleFactor),
      }
    }
    const listeners = new Set();
    const handler = e => {
      const { x, y } = getPos(e)
      ctx.fillStyle = 'black'
      ctx.fillRect(x*upscaleFactor, y*upscaleFactor,
                    2*halfPenSize*upscaleFactor, 2*halfPenSize*upscaleFactor)
      for (let yy = y - halfPenSize; yy < y + halfPenSize; yy++)
        for (let xx = x - halfPenSize; xx < x + halfPenSize; xx++)
          img[sz*Math.min(sz-1, Math.max(0, yy)) + Math.min(sz-1, Math.max(0, xx))] = 1
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        [...listeners].map(listener => listener(img))
      }, 100)
    }
    canvas.addEventListener('touchstart', e => {dragging=true; handler(e)})
    canvas.addEventListener('touchmove', e => {e.preventDefault(); dragging && handler(e)})
    canvas.addEventListener('touchend', () => dragging=false)
    canvas.addEventListener('mousedown', e => {dragging=true; handler(e)})
    canvas.addEventListener('mousemove', e => {dragging && handler(e)})
    canvas.addEventListener('mouseup', () => dragging=false)
    canvas.addEventListener('mouseleave', () => dragging=false)
    clear.addEventListener('click', () => {
      ctx.fillStyle = 'white'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      output.textContent = ''
      img = new Uint8Array(sz*sz)
    })
  
    return {
      onUpdate: listener => listeners.add(listener),
      showPreds: (values, indices) => {
        output.textContent = values.map(
            (v, i) => `${labels[indices[i]]} : ${v.toFixed(3)}`).join('\n')
      },
    };
  };