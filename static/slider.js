const speedThumb = document.querySelector('#speedthumb')
const speedTrack = document.querySelector('#speedtrack')
const speedText = document.querySelector("#speedtext");

const stepThumb = document.querySelector('#stepthumb')
const stepTrack = document.querySelector('#steptrack')
const stepText = document.querySelector("#steptext");

const updateSlider = (value, thumb, track, range, text) => {
  let percentage = (value / range.getAttribute("max")) * 100
  thumb.style.left = `${percentage}%`
  thumb.style.transform = `translate(-${percentage}%, -50%)`
  track.style.width = `${percentage}%`
  text.innerHTML = value;
}

speedSlider.oninput = (e) =>
  updateSlider(e.target.value, speedThumb, speedTrack, speedSlider, speedText);

stepSlider.oninput = (e) =>
  updateSlider(e.target.value, stepThumb, stepTrack, stepSlider, stepText);

updateSlider(24, speedThumb, speedTrack, speedSlider, speedText);
updateSlider(0, stepThumb, stepTrack, stepSlider, stepText);
