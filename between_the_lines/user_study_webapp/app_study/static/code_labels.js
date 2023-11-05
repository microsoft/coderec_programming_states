// load json file
var json_path = ""; //static/json_logs_7-03.json';
var video_path = ""; //static/july3-4pm-linearreg_accepts_trim.mp4";
var labeled_so_far = 0;
var logs_data;

// variables
var index_log = 0;

// html elements
var button = document.getElementById("pauseButton");
var button_autoplay = document.getElementById("autoPlay");
var autoplay_mode = false;
var video = document.getElementById("codeVideo");
var state_text = document.getElementById("state");
//var suggestion_text = document.getElementById("suggestion");
var state1_button = document.getElementById("state1");
var state2_button = document.getElementById("state2");
var state3_button = document.getElementById("state3");
var state4_button = document.getElementById("state4");
var state5_button = document.getElementById("state5");
var state6_button = document.getElementById("state6");
var state7_button = document.getElementById("state7");
var state8_button = document.getElementById("state8");
var state9_button = document.getElementById("state9");
var state10_button = document.getElementById("state10");
var state11_button = document.getElementById("state11");
var state12_button = document.getElementById("state12");
var stateIDK_button = document.getElementById("stateIDK");
var button_prev = document.getElementById("prev_index");
var button_next = document.getElementById("next_index");
var button_skip5sforward = document.getElementById("skip_5s_forward");
var button_skip5sbackward = document.getElementById("skip_5s_backward");
var speed_slider = document.getElementById("speed_slider");
var events_pannel = document.getElementById("events_pannel");
var button_replay = document.getElementById("replay_event");
var info_p = document.getElementById("info");
var labeled_so_far = 0;
var in_replay = false;
// Initialize the paths

function initialization() {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/initialize_html", true);
    xhr.setRequestHeader("Content-type", "application/json");
    xhr.onreadystatechange = function () {
        console.log(xhr.status);
        if (xhr.readyState === 4 && xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            json_path = json.json_path;
            video_path = json.video_path;
            labeled_so_far = json.labeled_so_far;
            console.log(json_path);
            console.log(video_path);
            // change video source
            video.src = json.video_path;
            fetch(json.json_path)
                .then((response) => {
                    return response.json();
                })
                .then((data) => {
                    logs_data = data;
                });
        }
    };
    xhr.send("nothing");
    console.log("sent");
}

initialization();

// VIDEO CONTROLS

button_autoplay.onclick = function () {
    in_replay = false;
    button_replay.innerHTML = "Replay Event";

    if (!in_replay) {
        if (autoplay_mode) {
            autoplay_mode = false;
            button_autoplay.innerHTML = "Auto Play";
            video.pause();
        } else {
            autoplay_mode = true;
            button_autoplay.innerHTML = "Stop Auto Play";
            video.currentTime = logs_data.logs[index_log-1].time + 0.001;
            video.play();
        }
    }

};

button.onclick = function () {
    return;
    var video = document.getElementById("codeVideo");
    var button = document.getElementById("pauseButton");
    if (video.paused) {
        video.play();
        button.innerHTML = "Pause";
    } else {
        video.pause();
        button.innerHTML = "Play";
    }
};

button_prev.onclick = function () {
    if (index_log > 0) {
        index_log--;
    }
    update_log_index();

    video.currentTime = logs_data.logs[index_log-1].time + 0.001;
    //video.pause();
    //in_replay = false;
    autoplay_mode = false;
    button_autoplay.innerHTML = "Auto Play";
    //button_replay.innerHTML = "Replay Event";
    // replay on each click
    in_replay = true;
    // replay video
    video.currentTime = logs_data.logs[index_log - 1].time + 0.001;
    video.play();
    button_replay.innerText = "Stop Replay";

};

button_next.onclick = function () {
    if (index_log < logs_data.logs.length - 1) {
        index_log++;
    }
    update_log_index();

    video.currentTime = logs_data.logs[index_log-1].time + 0.001;
    //video.pause();
    //in_replay = false;
    autoplay_mode = false;
    button_autoplay.innerHTML = "Auto Play";
    //button_replay.innerHTML = "Replay Event";
    // replay on each click
    in_replay = true;
    // replay video
    video.currentTime = logs_data.logs[index_log - 1].time + 0.001;
    video.play();
    button_replay.innerText = "Stop Replay";
};

button_skip5sbackward.onclick = function () {
    // not used
    video.currentTime = Math.max(0, video.currentTime - 5);
    // update index_log to match time
    for (var i = 0; i < logs_data.logs.length - 1; i++) {
        if (
            video.currentTime > logs_data.logs[i].time &&
            video.currentTime < logs_data.logs[i + 1].time
        ) {
            index_log = i;
            break;
        }
    }
    update_log_index();
};

button_skip5sforward.onclick = function () {
    // not used
    video.currentTime = Math.min(video.duration, video.currentTime + 5);
    for (var i = 0; i < logs_data.logs.length - 1; i++) {
        if (
            video.currentTime > logs_data.logs[i].time &&
            video.currentTime < logs_data.logs[i + 1].time
        ) {
            index_log = i;
            break;
        }
    }
    update_log_index();
};

speed_slider.oninput = function () {
    video.playbackRate = speed_slider.value;
};

button_replay.onclick = function () {
    // continuously replay the video, untill button is clicked
    // check in replay
    if (in_replay) {
        in_replay = false;
        button_replay.innerHTML = "Replay Event";
        video.pause();
    } else {
        in_replay = true;
        // replay video
        video.currentTime = logs_data.logs[index_log - 1].time + 0.001;
        video.play();
        button_replay.innerText = "Stop Replay";
    }
    autoplay_mode = false;

    button_autoplay.innerHTML = "Auto Play";

};
video.onseeked = function () {
    if (!in_replay) {
        for (var i = 1; i < logs_data.logs.length ; i++) {
            if (
                video.currentTime >= logs_data.logs[i-1].time &&
                video.currentTime <= logs_data.logs[i].time
            ) {
                index_log = i;
                break;
            }
        }
        update_log_index();
    }
};

// AUTO LOGS
var intervalId = window.setInterval(function () {
    // update progress bar
    var progress_bar = document.getElementById("myBar");
    // logs lenght
    var logs_length = logs_data.logs.length;
    var progress = (labeled_so_far / logs_length) * 100;
    progress = progress.toFixed(0);
    progress_bar.style.width = progress + "%";
    progress_bar.innerHTML = labeled_so_far + "/" + logs_length +"  (" +progress + "% Completed)";
    // length of logs
    // video current time
    var current_time = video.currentTime;
    // check if video paused
    if (video.paused) {
        // get element 
        var playpause_overlay = document.getElementById("playpause_overlay");
        // show overlay
        playpause_overlay.style.display = "block";
    }
    else{
        // hide overlay
        var playpause_overlay = document.getElementById("playpause_overlay");
        playpause_overlay.style.display = "none";
    }


    // check time of current log
    var new_index = 0;
    for (var i = 1; i < logs_data.logs.length ; i++) {
        if (
            video.currentTime >= logs_data.logs[i-1].time &&
            video.currentTime <= logs_data.logs[i].time
        ) {
            new_index = i;
            break;
        }
    }

    if (in_replay) {
        if (current_time > logs_data.logs[index_log].time) {
            video.currentTime = logs_data.logs[index_log - 1].time + 0.001;
        }
        return
    }

    if (autoplay_mode) {
        if (current_time < logs_data.logs[index_log ].time && current_time > logs_data.logs[index_log].time - 0.05) {
            video.pause();
            info_p.innerText = "Label The State Now";
        } 
    }

    if (new_index != index_log) {
        index_log = new_index;
        update_log_index();
    }
}, 1);

function update_log_index() {
    // get a string with all logs concatenated
    var logs_string = "";
    var bodydesc = document.getElementById("bodydesc");
    // scroll to the top
    bodydesc.scrollTop = 0;

    // get current state
    var current_state = logs_data.logs[index_log].HiddenState;
    var label_state = document.getElementById("label_state");
    if (current_state == "UserBeforeAction") {
        label_state.innerText = "What were you doing while the suggestion was being shown?";
        document.getElementById('suggestion_label').innerText = "Current Suggestion";

        // change opacity of state button 6 7 8 9 12
        state6_button.style.opacity = 0.2;
        state7_button.style.opacity = 0.2;
        state8_button.style.opacity = 0.2;
        state9_button.style.opacity = 0.2;
        state12_button.style.opacity = 0.2;

    }
    else{
        label_state.innerText = "What were you doing before the next suggestion was  shown?     ";
        document.getElementById('suggestion_label').innerText = "Next Suggestion";
        state6_button.style.opacity = 1;
        state7_button.style.opacity = 1;
        state8_button.style.opacity = 1;
        state9_button.style.opacity = 1;
        state12_button.style.opacity = 1;
    }

    var table = document.getElementById("table_events");
    table.innerHTML = '<tr> <th > ID </th> <th style="width:50%;">Timestamps</th> <th style="width:20%;">TimeInState</th> <th style="width:40%;">Label</th> </tr> ';
    
    for (var i = 0; i < logs_data.logs.length; i++) {
        //if (i <= index_log - 3) {
        //    bodydesc.scrollTop += 30;
        //}

        var row = table.insertRow();
        var time_in_state = logs_data.logs[i].time - logs_data.logs[Math.max(i - 1,0)].time;
        var time_in_state_string = time_in_state.toFixed(2);
        if (i == index_log) {
            // text highlighted for current log

            var cell0 = row.insertCell();
            var cell1 = row.insertCell();
            var cell2 = row.insertCell();
            var cell3 = row.insertCell();
            cell0.innerHTML = i;
            cell1.innerHTML = "[" + logs_data.logs[Math.max(0, i - 1)].time.toFixed(2) + "," + logs_data.logs[i].time.toFixed(2) + "]";
            cell2.innerHTML = time_in_state_string;
            cell3.innerHTML = logs_data.logs[i].label.replace(/ /g,"_");;
        } else {
            var cell0 = row.insertCell();
            var cell1 = row.insertCell();
            var cell2 = row.insertCell();
            var cell3 = row.insertCell();
            cell0.innerHTML = i;
            cell1.innerHTML = "[" + logs_data.logs[Math.max(0, i - 1)].time.toFixed(2) + "," + logs_data.logs[i].time.toFixed(2) + "]";
            cell2.innerHTML = time_in_state_string;
            cell3.innerHTML = logs_data.logs[i].label.replace(/ /g,"_");
        }
    }

    // line is zero-based
    // line is the row number that you want to see into view after scroll  

    var rows = table.querySelectorAll('tr');
    
    rows.forEach(row => row.classList.remove('active'))
    rows[index_log+1].classList.add('active');
    rows[index_log+1].scrollIntoView({
    	//behavior: 'smooth',
      block: 'center'
    });
    //suggestion_text.innerText = logs_data.logs[index_log].CurrentSuggestion;
    state_text.innerText = logs_data.logs[index_log].CurrentSuggestion;//logs_data.logs[index_log].label;
    // check if statename is shown and update the innertext of statebuttons
    //if (logs_data.logs[index_log].StateName == "Shown") {
    //    state_button.innerText = "No state";
    //}
/*     if (logs_data.logs[index_log].HiddenState == "UserBeforeAction") {
        state6_button.disabled = true;
        state7_button.disabled = true;
        state8_button.disabled = true;
        state9_button.disabled = true;
    }
    if (logs_data.logs[index_log].HiddenState == "UserTypingOrPaused") {
        state6_button.disabled = false;
        state7_button.disabled = false;
        state8_button.disabled = false;
        state9_button.disabled = false;
    }
    if (logs_data.logs[index_log].HiddenState == "UserTyping") {
        state6_button.disabled = false;
        state7_button.disabled = false;
        state8_button.disabled = false;
        state9_button.disabled = false;
    }
    if (logs_data.logs[index_log].HiddenState == "UserPaused") {
        state6_button.disabled = false;
        state7_button.disabled = false;
        state8_button.disabled = false;
        state9_button.disabled = false;
    } */
    
}

function update_json_file() {
    // check if at indext the label is not_labeled
    if (autoplay_mode) {
        video.currentTime = logs_data.logs[index_log].time + 0.001;
        info_p.innerText = "";
        video.play();
    }
    var json_path = "static/json_logs_7-03.json";
    var json_data = JSON.stringify(logs_data);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/update_json", true);
    xhr.setRequestHeader("Content-type", "application/json");
    xhr.onreadystatechange = function () {
        console.log(xhr.status);
    };
    xhr.send(json_data);
    console.log("sent");
}

// LABELING OPTIONS
var custom_state = document.getElementById("custom_state");
var save_state = document.getElementById("save_state");

save_state.onclick = function () {
    if (custom_state.value == "") {
        alert("Please enter a state name");
        return;
    }
    if (logs_data.logs[index_log].label == "not_labeled") {
        // update label
        labeled_so_far++;
    }
    // get from input the label
    logs_data.logs[index_log].label = custom_state.value;
    // save json file
    update_json_file();
    update_log_index();
    custom_state.value = "";
}

function log_state_button(state_text){
    if (logs_data.logs[index_log].label == "not_labeled") {
        // update label
        labeled_so_far++;
    }
    logs_data.logs[index_log].label = state_text;
    // save json file
    update_json_file();
    update_log_index();
}


state1_button.onclick = function () {
    log_state_button(state1_button.innerText);
};

state2_button.onclick = function () {
    log_state_button(state2_button.innerText);
};

state3_button.onclick = function () {
    log_state_button(state3_button.innerText);

};

state4_button.onclick = function () {
    log_state_button(state4_button.innerText);
};

state5_button.onclick = function () {
    log_state_button(state5_button.innerText);
};

state6_button.onclick = function () {
    log_state_button(state6_button.innerText);
}

state7_button.onclick = function () {
    log_state_button(state7_button.innerText);
}

state8_button.onclick = function () {
    log_state_button(state8_button.innerText);
}
state9_button.onclick = function () {
    log_state_button(state9_button.innerText);
}
state10_button.onclick = function () {
    log_state_button(state10_button.innerText);
}
state11_button.onclick = function () {
    log_state_button(state11_button.innerText);
}
state12_button.onclick = function () {
    log_state_button(state12_button.innerText);
}

stateIDK_button.onclick = function () {
    log_state_button(stateIDK_button.innerText);
}


// keyboard shortcuts

// on right arrow
document.onkeydown = function (e) {
    // check if user is inputting a text
    if (custom_state.value != "") {
        return;
    }

    if (e.keyCode == 39) {
        button_next.click();
    }

    if (e.keyCode == 37) {
        button_prev.click();
    }

    // a
    if (e.keyCode == 65) {
        state1_button.click();
    }
    // s
    if (e.keyCode == 83) {
        state2_button.click();
    }
    // d
    if (e.keyCode == 68) {
        state3_button.click();
    }
    // f
    if (e.keyCode == 70) {
        state4_button.click();
    }
    // g
    if (e.keyCode == 71) {
        state5_button.click();
    }
    // z
    if (e.keyCode == 90) {
        state6_button.click();
    }
    // x
    if (e.keyCode == 88) {
        state7_button.click();
    }
    // c
    if (e.keyCode == 67) {
        state8_button.click();
    }
    // v
    if (e.keyCode == 86) {
        state9_button.click();
    }
    // b
    if (e.keyCode == 66) {
        state12_button.click();
    }
    // h
    if (e.keyCode == 72) {
        state11_button.click();
    }
    // n   
    if (e.keyCode == 78) {
        state10_button.click();
    }

    // r button
    if (e.keyCode == 82) {
        // replay button
        button_replay.click();
    }
    // I   
    if (e.keyCode == 73) {
        stateIDK_button.click();
    }

    // space
    if (e.keyCode == 32) {
        button_autoplay.click();
    }
    

};

var popup = document.getElementById("popup");
popup.onclick = function () {
    var myPopup = document.getElementById("myPopup");
    myPopup.classList.toggle("show");
}

