
// Video recording
const timer = document.getElementById("timer");
let loadingFeedback = document.getElementById("loading-feedback");
const audio = document.getElementById("audio");

// video recording part
window.onload = async () => {
    if (timer) {
        audio.play();
        await updateTimer();
    } else if (loadingFeedback) {
        audio.pause();
        await pollLoading();
    }
};


async function updateTimer() {
    let seconds = 10
    while (seconds > 0) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        seconds--;
        timer.innerText = "00:0"+seconds
    }
    let form = new FormData();
    form.append("rec", '');
    let response = await fetch("/requests", {method: "POST", body: form});
    document.body.innerHTML = await response.text();
    await pollLoading();
}


async function pollLoading() {
    loadingFeedback = document.getElementById("loading-feedback");
    let progressBar = document.getElementById("progress");
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        fetch("/poll").then((response) => {
            response.json().then((data) => {
                if (data[1]) {
                    loadingFeedback.innerText = data[0] + "/" + data[1] + " frames";
                    progressBar.ariaValueNow = data[0]
                    progressBar.ariaValueMax = data[1]
                    progressBar.setAttribute("style", "width: " + (data[0]/data[1])*100 + "%")
                    if (data[0] === data[1]) {
                        window.location.assign("/results?keypoints=true");
                    }
                }
            });
        });
    }
}

function setMusicName(name, index) {
    document.getElementById("songname").innerText = name;
    document.getElementById("songindex").innerText = index;
    document.getElementById("recordingbutton").removeAttribute("disabled");
    document.getElementById("recordingbutton").classList.remove("disabled");
    audio.innerHTML = "";
    let src = document.createElement('source');
    src.src = '/music/' + index
    src.type = "audio/wav"
    audio.appendChild(src);
}