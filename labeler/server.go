package main

import (
	"fmt"
	"html/template"
	"net/http"
)

func main() {
	inputFolder := "/home/david/Dropbox/Journal"
	outputFolder := "/home/david/Dropbox/Journal/Labels"
	name := "2019-07-24.jpg"

	_, err := NewLabeler(name, inputFolder, outputFolder)
	if err != nil {
		panic(err)
	}

	indexTmpl := template.Must(template.ParseFiles("./index.html"))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		err := indexTmpl.Execute(w, nil)
		if err != nil {
			_ = fmt.Errorf("error executing template, %s", err.Error())
		}
	})
	// TODO: Handlers for interacting with the labeler

	err = http.ListenAndServe(":8080", nil)
	if err != nil {
		panic(err)
	}
}
