package main

import (
	"bytes"
	"fmt"
	"html/template"
	"image/jpeg"
	"net/http"
	"os"
	"path"
	"strconv"

	"github.com/pkg/errors"
)

func main() {
	inputFolder := "/home/david/Dropbox/Journal"
	outputFolder := "/home/david/Dropbox/Journal/Labels"
	name := "2019-07-24.jpg"

	l, err := NewLabeler(name, inputFolder, outputFolder)
	if err != nil {
		panic(err)
	}

	indexTmpl := template.Must(template.ParseFiles("./index.html"))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		possibleID := path.Base(r.URL.Path)
		firstUnlabeled := l.FirstUnlabeled()
		var curIdx int
		if len(possibleID) > 0 {
			var err error
			var curIdxRaw int64
			curIdxRaw, err = strconv.ParseInt(possibleID, 10, 64)
			if err != nil {
				http.Error(w, "invalid word ID", 400)
				return
			}
			curIdx = int(curIdxRaw)
		} else {
			if firstUnlabeled != -1 {
				curIdx = firstUnlabeled
			}
		}
		prevIdx := curIdx - 1
		if prevIdx < 0 {
			prevIdx = 0
		}
		i, err := l.Image(curIdx)
		if err != nil {
			http.Error(w, errors.Wrap(err, "invalid word ID").Error(), 400)
			return
		}

		data := IndexData{
			HasUnlabeled:      firstUnlabeled != -1,
			FirstUnlabeledIdx: firstUnlabeled,
			TotalWords:        l.Len(),
			NextIdx:           curIdx + 1,
			PrevIdx:           prevIdx,
			CurImageIdx:       curIdx,
			CurLabel:          i.label,
			CurImageURL:       fmt.Sprintf("/word/%d", curIdx),
		}

		err = indexTmpl.Execute(w, data)
		if err != nil {
			_ = fmt.Errorf("error executing template, %s", err.Error())
		}
	})
	http.HandleFunc("/word/", func(w http.ResponseWriter, r *http.Request) {
		possibleID := path.Base(r.URL.Path)
		curIdxRaw, err := strconv.ParseInt(possibleID, 10, 64)
		if err != nil {
			http.Error(w, "invalid word ID", 400)
			return
		}
		i, err := l.Image(int(curIdxRaw))
		if err != nil {
			http.Error(w, errors.Wrap(err, "unable to load image").Error(), 400)
			return
		}
		image, err := i.Mat.ToImage()
		if err != nil {
			http.Error(w, errors.Wrap(err, "unable to convert image").Error(), 500)
			return
		}

		buffer := new(bytes.Buffer)
		if err := jpeg.Encode(buffer, image, nil); err != nil {
			http.Error(w, errors.Wrap(err, "unable to encode image as JPEG").Error(), 500)
			return
		}

		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("Content-Length", strconv.Itoa(len(buffer.Bytes())))
		if _, err := w.Write(buffer.Bytes()); err != nil {
			http.Error(w, errors.Wrap(err, "unable to output image").Error(), 500)
			return
		}
	})
	http.HandleFunc("/save_label", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "wrong method", 405)
			return
		}
		if err := r.ParseForm(); err != nil {
			http.Error(w, errors.Wrap(err, "unable to parse form").Error(), 400)
			return
		}
		curIdxRaw, err := strconv.ParseInt(r.FormValue("idx"), 10, 64)
		if err != nil {
			http.Error(w, "invalid word ID", 400)
			return
		}
		err = l.LabelWord(int(curIdxRaw), r.FormValue("label"))
		if err != nil {
			http.Error(w, errors.Wrap(err, "unable to label word").Error(), 400)
			return
		}

		// We've labeled a word, time to go to the next one
		if l.FirstUnlabeled() < int(curIdxRaw) && int(curIdxRaw) < l.Len()-1 {
			// They were correcting some mistakes maybe, just go to the next index
			http.Redirect(w, r, fmt.Sprintf("/%d", curIdxRaw+1), 303)
		} else if l.FirstUnlabeled() != -1 {
			http.Redirect(w, r, fmt.Sprintf("/%d", l.FirstUnlabeled()), 303)
		} else {
			// Everything is now labeled!
			http.Redirect(w, r, fmt.Sprintf("/%d", curIdxRaw), 303)
		}
	})

	http.HandleFunc("/save_all", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "wrong method", 405)
			return
		}
		err := l.SaveAndClose(outputFolder)
		if err != nil {
			http.Error(w, errors.Wrap(err, "unable to save labels").Error(), 500)
			return
		}
		// Our work here is done!
		os.Exit(0)
	})

	// TODO: Handlers for interacting with the labeler

	err = http.ListenAndServe(":8080", nil)
	if err != nil {
		panic(err)
	}
}

type IndexData struct {
	HasUnlabeled      bool
	FirstUnlabeledIdx int
	TotalWords        int
	NextIdx           int
	PrevIdx           int

	CurImageIdx int
	CurLabel    string
	CurImageURL string
}
