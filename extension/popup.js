$(document).ready(function () {
    $("button").click(function () {
        // Show loading message
        $("#div1").text("Loading...");

        var tablink;
        chrome.tabs.getSelected(null, function (tab) {
            tablink = tab.url;
            $("#p1").text("The URL being tested is - " + tablink);

            var xhr = new XMLHttpRequest();
            var params = "url=" + tablink;
            xhr.open("POST", "http://localhost/php_program/clientServer.php", false);
            xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
            xhr.send(params);

            // Extract the response text starting from "Predicted result: bad"
            var responseText = xhr.responseText;
            var startIndex = responseText.indexOf("Predicted result: bad");
            if (startIndex !== -1) {
                responseText = responseText.substring(startIndex);
            }

            // Update the div with the response text
            $("#div1").text(responseText);
        });
    });

    chrome.tabs.getSelected(null, function (tab) {
        var tablink = tab.url;
        $("#p1").text("The URL being tested is - " + tablink);
    });
});
