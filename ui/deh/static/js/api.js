function get_backend_config() {
    config = document.getElementById("config")

    axios.get('/')
        .then(function (response) {
            var config_details = response.data;
            console.log(config_details);
            config.innerHTML = "<pre>" + JSON.stringify(config_details, null, 2) + "</pre>"
        });
}