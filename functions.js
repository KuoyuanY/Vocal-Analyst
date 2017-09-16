function HandleBrowseClick(c)
{
	if (c == 'r')
    	var fileinput = document.getElementById("rbrowse");
    else
    	var fileinput = document.getElementById("lbrowse");
    fileinput.click();
}

function Handlechange(c)
{
	if (c == 'r') {
    	var fileinput = document.getElementById("rbrowse");
    	var textinput = document.getElementById("rfilename");
    } else {
    	var fileinput = document.getElementById("lbrowse");
    	var textinput = document.getElementById("lfilename");
    }
    textinput.value = fileinput.value;
}

function CompareAndPlayMusic() {
	Compare();
	playMusic();
}

function Compare() {
	window.open('file:///C:/Users/power/Documents/BRH/index.html','_self',false);
}

function playMusic() {
	var audio = new Audio(document.getElementById("lfilename"));
	audio.play();
}