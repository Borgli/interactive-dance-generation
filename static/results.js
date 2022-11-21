const fileReader = document.getElementById("file");
const speedSlider = document.getElementById("speed");
const stepSlider = document.getElementById("step");
const replayButton = document.getElementById("replay");
const startIcon = document.getElementById("start");
const stopIcon = document.getElementById("stop");
const playButton = document.getElementById("play");

let timeout = speedSlider.value;

fileReader.addEventListener("change", async e => await readFile(fileReader));

speedSlider.addEventListener("input", e => {
    timeout = speedSlider.value;
    speedText.innerHTML = speedSlider.value;
});

stepSlider.addEventListener("input", e => {
    if (player.isRunning) {
        player.step = stepSlider.value;
    } else {
        player.runStep(stepSlider.value);
    }
    stepText.innerHTML = stepSlider.value;
});

replayButton.addEventListener("click", async e => {
    await player.stop();
    await player.play(0);
});

playButton.addEventListener("click", async e => {
    if (player.isRunning) {
        startIcon.classList.remove("visually-hidden")
        stopIcon.classList.add("visually-hidden");
        await player.stop();
    } else {
        startIcon.classList.add("visually-hidden");
        stopIcon.classList.remove("visually-hidden");
        await player.stop();
        await player.play();
    }
});

window.onload = async () => {
    let urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('keypoints')) {
        let response = await fetch('/get_keypoints');
        let data = await response.json();

        replayButton.classList.remove("disabled");
        replayButton.removeAttribute("disabled");

        playButton.classList.remove("disabled");
        playButton.removeAttribute("disabled");

        startIcon.classList.add("visually-hidden");
        stopIcon.classList.remove("visually-hidden");

        player.data = data;
        await player.play();
    }
};

d3.select("#svgContainer").append("svg")
    .attr("id", "mainsvg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", "0 0 1920 1080");

class Player {

    data = [];
    continue = true;
    radius = 8;
    color = "purple";
    isRunning = false;
    step = 0;

    set data(data) {
        this.data = data;
    }

    constructor() {
        this.svg = d3.select("#mainsvg");
    }

    async stop() {
        this.continue = false;
        while (this.isRunning) {
            await new Promise(resolve => setTimeout(resolve, timeout));
        }
    }

    runStep(step) {
        this.step = step
        this.writeCircle();
    }

    async runPlayer(i = this.step) {
        this.continue = true;
        this.step = i
        stepSlider.setAttribute("max", this.data.length - 1);
        stepSlider.value = this.step % this.data.length
        updateSlider(stepSlider.value, stepThumb, stepTrack, stepSlider, stepText);
        while (this.continue) {
            this.writeCircle()
            await new Promise(resolve => setTimeout(resolve, timeout));
            this.step++;
            stepSlider.value = this.step % this.data.length
            updateSlider(stepSlider.value, stepThumb, stepTrack, stepSlider, stepText);
        }
    }

    writeCircle() {
        this.svg.selectAll("circle")
                .data(this.data[this.step % this.data.length])
                .join("circle")
                .attr("cx", (d) => d[0]*1920/2 + 1920/2)
                .attr("cy", (d) => d[1]*1080/2 + 1080/2)
                .attr("r", this.radius)
                .style("fill", this.color)
    }

    async play(i = this.step) {
        if (this.data.length > 0) {
            this.isRunning = true;
            await this.runPlayer(i);
            this.isRunning = false;
        }
    }
}

const player = new Player();

async function readFile(input) {
    let file = input.files[0];

    let reader = new FileReader();

    reader.readAsText(file);

    reader.onload = async function() {
        replayButton.classList.remove("disabled");
        replayButton.removeAttribute("disabled");

        playButton.classList.remove("disabled");
        playButton.removeAttribute("disabled");

        startIcon.classList.add("visually-hidden");
        stopIcon.classList.remove("visually-hidden");

        player.data = JSON.parse(reader.result);
        await player.play()
    };

    reader.onerror = function() {
        console.log(reader.error);
    };
}