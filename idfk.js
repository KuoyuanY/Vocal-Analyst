var song_name = "Kings Never Die";
var artist_name = "Eminem";
window.open("https://www.youtube.com/results?search_query=" + song_name + " - " + artist_name);
$.get("https://www.youtube.com/results?search_query=" + song_name + " - " + artist_name, function(response) { 
    alert(response) 
});