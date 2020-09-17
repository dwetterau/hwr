package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/pkg/errors"

	"github.com/dwetterau/hwr/word_finder"
)

type Labeler struct {
	name     string
	allWords []ImgAndLabel

	sync.Mutex
}

func NewLabeler(name string, inputFolder, outputFolder string) (*Labeler, error) {
	existingLabels := readLabelFile(name, outputFolder)
	filename := filepath.Join(inputFolder, name)
	foundWords, err := word_finder.Find(filename)
	if err != nil {
		return nil, err
	}

	finalImages := make([]ImgAndLabel, len(foundWords))
	// Fill in any existing labels
	for i := range finalImages {
		finalImages[i].ImgAndReference = foundWords[i]
		word := finalImages[i]
		if i < len(existingLabels) {
			if existingLabels[i].Line != word.Line {
				continue
			}
			if existingLabels[i].Word != word.Word {
				continue
			}
			finalImages[i].label = existingLabels[i].label
			finalImages[i].hasLabel = true
		}
	}
	return &Labeler{
		name:     name,
		allWords: finalImages,
	}, nil
}

func (l *Labeler) Len() int {
	return len(l.allWords)
}

func (l *Labeler) NumUnlabeled() int {
	l.Lock()
	defer l.Unlock()

	n := 0
	for _, w := range l.allWords {
		if !w.hasLabel {
			n++
		}
	}
	return n
}

// Returns -1 if no unlabeled image is found
func (l *Labeler) FirstUnlabeled() int {
	l.Lock()
	defer l.Unlock()
	for i, img := range l.allWords {
		if !img.hasLabel {
			return i
		}
	}
	return -1
}

func (l *Labeler) Image(i int) (ImgAndLabel, error) {
	l.Lock()
	defer l.Unlock()
	if i > len(l.allWords) {
		return ImgAndLabel{}, errors.New(fmt.Sprintf("invalid current word: %d", i))
	}
	return l.allWords[i], nil
}

func (l *Labeler) LabelWord(i int, label string) error {
	l.Lock()
	defer l.Unlock()

	if i > len(l.allWords) {
		return errors.New(fmt.Sprintf("invalid current word: %d", i))
	}
	l.allWords[i].hasLabel = true
	l.allWords[i].label = label
	return nil
}

func (l *Labeler) SaveAndClose(outputFolder string) error {
	l.Lock()
	defer l.Unlock()

	err := saveLabels(l.name, outputFolder, l.allWords)
	for _, i := range l.allWords {
		_ = i.Mat.Close()
	}
	return err
}

/*
	// open display window
	window := gocv.NewWindow("HWR")
	defer window.Close()


	for firstUnlabeled < len(finalImages) {
		toDraw := finalImages[firstUnlabeled]
		fmt.Printf("Labeling %d,%d - %t: %s\n", toDraw.Line, toDraw.Word, toDraw.hasLabel, toDraw.label)

		c := origImg.Clone()
		gocv.Rectangle(&c, image.Rect(
			toDraw.OrigC,
			toDraw.OrigR,
			toDraw.OrigC+toDraw.mat.Cols(),
			toDraw.OrigR+toDraw.mat.Rows(),
		), color.RGBA{0, 0, 255, 0}, 3)

		m := sync.Mutex{}
		running := true
		go func() {
			scanner := bufio.NewScanner(os.Stdin)
			if scanner.Scan() {
				t := scanner.Text()
				m.Lock()
				r := running
				m.Unlock()
				if r {
					finalImages[firstUnlabeled].label = t
					finalImages[firstUnlabeled].hasLabel = true
					m.Lock()
					running = false
					m.Unlock()
				} else {
					fmt.Println("Dropping input..")
				}
			}
		}()
		window.IMShow(c)
		shouldStop := false
		for {
			m.Lock()
			r := running
			m.Unlock()
			if !r {
				break
			}
			key := window.WaitKey(100)
			if key == 27 {
				fmt.Println("exit pressed, stopping early")
				m.Lock()
				running = false
				m.Unlock()
				shouldStop = true
				break
			}
			// prev
			if key == 2 || key == 81 {
				m.Lock()
				running = false
				m.Unlock()
				firstUnlabeled -= 2
				fmt.Println("waiting to drop input..")
				break
			}
			// next
			if key == 3 || key == 83 {
				m.Lock()
				running = false
				m.Unlock()
				fmt.Println("waiting to drop input..")
				break
			}
		}
		_ = c.Close()
		if shouldStop {
			break
		}
		firstUnlabeled++
	}

*/

func labelFilePath(name, outputFolder string) string {
	return filepath.Join(outputFolder, name[:len(name)-len(filepath.Ext(name))]+".csv")
}

// File format:
// r,c,w,h,l,w,label
func readLabelFile(name, outputFolder string) []ImgAndLabel {
	labelFile, err := ioutil.ReadFile(labelFilePath(name, outputFolder))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		panic(err)
	}
	lines := strings.Split(strings.TrimSpace(string(labelFile)), "\n")
	existing := make([]ImgAndLabel, 0, len(lines))
	for _, l := range lines {
		if len(l) == 0 {
			continue
		}
		sl := strings.Split(l, ",")
		mustParseIndex := func(i int) int {
			num, err := strconv.ParseInt(sl[i], 10, 64)
			if err != nil {
				panic(err)
			}
			return int(num)
		}

		existing = append(existing, ImgAndLabel{
			ImgAndReference: word_finder.ImgAndReference{
				OrigR:  mustParseIndex(0),
				OrigC:  mustParseIndex(1),
				Width:  mustParseIndex(2),
				Height: mustParseIndex(3),
				Line:   mustParseIndex(4),
				Word:   mustParseIndex(5),
			},
			label:    strings.Join(sl[6:], ","),
			hasLabel: true,
		})
	}
	return existing
}

func saveLabels(name, outputFolder string, labels []ImgAndLabel) error {
	lines := make([]string, 0, len(labels))
	for _, l := range labels {
		if !l.hasLabel {
			continue
		}
		lines = append(lines, l.csvLine())
	}
	return ioutil.WriteFile(
		labelFilePath(name, outputFolder),
		[]byte(strings.Join(lines, "\n")),
		0666,
	)
}

type ImgAndLabel struct {
	word_finder.ImgAndReference
	label    string
	hasLabel bool
}

func (i ImgAndLabel) csvLine() string {
	return fmt.Sprintf(
		"%d,%d,%d,%d,%d,%d,%s",
		i.OrigR,
		i.OrigC,
		i.Width,
		i.Height,
		i.Line,
		i.Word,
		i.label,
	)
}
