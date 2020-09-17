package word_finder

import (
	"errors"
	"image"
	"sort"

	"gocv.io/x/gocv"
)

// TODO:
// - Pipe each word into a ML model: https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5 / https://github.com/githubharald/SimpleHTR

type ImgAndReference struct {
	Mat           gocv.Mat
	OrigR, OrigC  int
	Width, Height int
	Line, Word    int
}

func Find(imgFilename string) ([]ImgAndReference, error) {
	origImg := gocv.IMRead(imgFilename, gocv.IMReadGrayScale)
	if origImg.Empty() {
		return nil, errors.New("didn't load image")
	}
	lineImages := detectLines(origImg)

	finalImages := make([]ImgAndReference, 0, len(lineImages)*10)
	for _, i := range lineImages {
		finalImages = append(finalImages, detectWordsInternal(i)...)
	}
	return finalImages, origImg.Close()
}

func detectWordsInternal(orig ImgAndReference) []ImgAndReference {
	origImg := orig.Mat
	blurred := gocv.NewMat()
	gocv.Blur(origImg, &blurred, image.Point{X: 5, Y: 5})

	thresholded := gocv.NewMat()
	defer func() { _ = thresholded.Close() }()
	gocv.AdaptiveThreshold(blurred, &thresholded, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 11)
	_ = blurred.Close()

	dilated := gocv.NewMat()
	dilationKernal := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 25, Y: 3})
	gocv.Dilate(thresholded, &dilated, dilationKernal)

	finalBinaryImage := dilated

	contours := gocv.FindContours(finalBinaryImage, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	_ = finalBinaryImage.Close()
	rectangles := make([]image.Rectangle, 0, len(contours))
	for _, c := range contours {
		r := gocv.BoundingRect(c)
		s := r.Size()
		// Exclude noise around the edges
		if r.Max.Y == origImg.Rows() {
			if r.Min.Y > origImg.Rows()/2 {
				continue
			}
		}
		if r.Min.Y == 0 {
			if r.Max.Y < origImg.Rows()/2 {
				continue
			}
		}
		if s.X > 50 && s.Y > 15 {
			rectangles = append(rectangles, r)
		}
	}
	sort.Slice(rectangles, func(i, j int) bool {
		if rectangles[i].Min.X != rectangles[j].Min.X {
			return rectangles[i].Min.X < rectangles[j].Min.X
		}
		if rectangles[i].Min.Y != rectangles[j].Min.Y {
			return rectangles[i].Min.Y < rectangles[j].Min.Y
		}
		return rectangles[i].Size().X < rectangles[j].Size().X
	})

	toOutput := make([]ImgAndReference, len(rectangles))
	for i, rect := range rectangles {
		s := rect.Size()
		mat := gocv.NewMatWithSize(s.Y, s.X, thresholded.Type())
		for r := 0; r < s.Y; r++ {
			for c := 0; c < s.X; c++ {
				mat.SetUCharAt(r, c, 255-thresholded.GetUCharAt(rect.Min.Y+r, rect.Min.X+c))
			}
		}
		toOutput[i] = ImgAndReference{
			Mat:    mat,
			OrigR:  orig.OrigR + rect.Min.Y,
			OrigC:  orig.OrigC + rect.Min.X,
			Width:  mat.Cols(),
			Height: mat.Rows(),
			Line:   orig.Line,
			Word:   i,
		}
	}
	return toOutput
}

func detectLines(origImg gocv.Mat) []ImgAndReference {

	blurred := gocv.NewMat()
	gocv.Blur(origImg, &blurred, image.Point{X: 5, Y: 5})

	thresholded := gocv.NewMat()
	gocv.AdaptiveThreshold(blurred, &thresholded, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 11)
	_ = blurred.Close()

	dilated := gocv.NewMat()
	dilationKernal := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 9, Y: 1})
	gocv.Dilate(thresholded, &dilated, dilationKernal)
	_ = thresholded.Close()

	invertedDilated := gocv.NewMat()
	gocv.BitwiseNot(dilated, &invertedDilated)
	_ = dilated.Close()

	finalBinaryImage := invertedDilated

	// Generate chunks
	bigChunk := &chunkStruct{
		index:      0,
		startPixel: 0,
		chunkWidth: finalBinaryImage.Cols(),
		mat:        finalBinaryImage,
	}

	// Get the initial lines
	idToValleys := bigChunk.findPeaksAndValleys()
	positions := make([]int, 0, len(bigChunk.valleyIDs))
	for _, valleyID := range bigChunk.valleyIDs {
		v := idToValleys[valleyID]

		y := v.position
		positions = append(positions, y)

		//pt1 := image.Pt(0, y)
		//pt2 := image.Pt(origImg.Cols(), y)
		//gocv.Line(&origImg, pt1, pt2, color.RGBA{0, 255, 0, 50}, 3)
	}

	sort.Slice(positions, func(i, j int) bool {
		return positions[i] < positions[j]
	})
	// Add an above and below line for good measure
	minPosition := positions[0]
	maxPosition := positions[len(positions)-1]

	totalDeltaSum := 0
	for i := 1; i < len(positions); i++ {
		totalDeltaSum += positions[i] - positions[i-1]
	}
	avgHeight := totalDeltaSum / (len(positions) - 1)

	if minPosition > avgHeight {
		y := minPosition - avgHeight
		//pt1 := image.Pt(0, y)
		//pt2 := image.Pt(origImg.Cols(), y)
		//gocv.Line(&origImg, pt1, pt2, color.RGBA{0, 255, 0, 50}, 3)
		positions = append([]int{y}, positions...)
	}
	if maxPosition+avgHeight < origImg.Rows() {
		y := maxPosition + avgHeight
		//pt1 := image.Pt(0, y)
		//pt2 := image.Pt(origImg.Cols(), y)
		//gocv.Line(&origImg, pt1, pt2, color.RGBA{0, 255, 0, 50}, 3)
		positions = append(positions, y)
	}

	start := 0
	if positions[0]-avgHeight > 0 {
		start = positions[0] - avgHeight
	}
	toOutput := make([]ImgAndReference, 0, len(positions)+1)
	for i := 0; i <= len(positions); i++ {
		end := 0
		if i == len(positions) {
			end = positions[i-1] + avgHeight
		} else {
			end = positions[i]
		}
		dest := gocv.NewMatWithSize(end-start, origImg.Cols(), origImg.Type())
		for c := 0; c < origImg.Cols(); c++ {
			for r := start; r < end; r++ {
				dest.SetUCharAt(r-start, c, origImg.GetUCharAt(r, c))
			}
		}
		toOutput = append(toOutput, ImgAndReference{
			Mat:    dest,
			OrigR:  start,
			OrigC:  0,
			Width:  dest.Cols(),
			Height: dest.Rows(),
			Line:   i,
		})
		start = end
	}
	_ = finalBinaryImage.Close()

	return toOutput
}

const (
	valleyFactor      = .60
	expectedAvgHeight = 94
	blackThreshold    = 0
)

type valleyID int

type chunkStruct struct {
	index      int
	startPixel int
	chunkWidth int
	mat        gocv.Mat
	histogram  []int
	peaks      []peakStruct
	valleyIDs  []valleyID

	linesCount int
}

func (c *chunkStruct) avgHeight() int {
	diffs := 0
	for i := 1; i < len(c.peaks); i++ {
		prev := c.peaks[i-1].position
		cur := c.peaks[i].position
		diffs += cur - prev
	}
	if len(c.peaks) < 2 {
		return expectedAvgHeight
	}
	guess := diffs / (len(c.peaks) - 1)
	if guess < expectedAvgHeight/2 || guess > expectedAvgHeight*1.5 {
		return expectedAvgHeight
	}
	return guess
}

func (c *chunkStruct) findPeaksAndValleys() map[valleyID]*valleyStruct {
	valleyIDCounter := valleyID(0)
	idToValley := make(map[valleyID]*valleyStruct, 50)
	c.calculateHistogram()

	for i := 1; i < len(c.histogram)-1; i++ {
		leftVal := c.histogram[i-1]
		centerVal := c.histogram[i]
		rightVal := c.histogram[i+1]
		avgHeight := c.avgHeight()

		// See if we're looking at a peak
		if centerVal >= leftVal && centerVal >= rightVal {
			if len(c.peaks) > 0 {
				last := c.peaks[len(c.peaks)-1]
				if (i-last.position) <= avgHeight/2 && centerVal >= last.value {
					c.peaks[len(c.peaks)-1].position = i
					c.peaks[len(c.peaks)-1].value = centerVal
				} else if (i-last.position) <= avgHeight/2 && centerVal < last.value {
					// Pretend it's not a peak
				} else {
					c.peaks = append(c.peaks, peakStruct{position: i, value: centerVal})
				}
			} else {
				c.peaks = append(c.peaks, peakStruct{position: i, value: centerVal})
			}
		}
	}

	peaksAverageValues := 0
	for _, p := range c.peaks {
		peaksAverageValues += p.value
	}
	if len(c.peaks) > 0 {
		peaksAverageValues /= len(c.peaks)
	}

	newPeaks := make([]peakStruct, 0, len(c.peaks))
	for _, p := range c.peaks {
		if p.value >= int(float64(peaksAverageValues)*valleyFactor) {
			newPeaks = append(newPeaks, p)
		}
	}

	/*
		if c.index >= totalChunks-1 {
			fmt.Println(len(c.peaks))
			for _, p := range c.peaks {
				fmt.Println(p.position, p.value)
				gocv.Line(&c.mat, image.Point{X: 0, Y: p.position}, image.Point{X: c.chunkWidth - 1, Y: p.position}, color.RGBA{255, 0, 0, 0}, 2)
			}
		}
	*/

	c.linesCount = len(newPeaks)
	c.peaks = newPeaks
	sort.Slice(c.peaks, func(i, j int) bool {
		return c.peaks[i].position < c.peaks[j].position
	})
	avgHeight := c.avgHeight()

	for i := 1; i < len(c.peaks); i++ {
		minPosition := (c.peaks[i-1].position + c.peaks[i].position) / 2
		minValue := c.histogram[minPosition]

		for j := c.peaks[i-1].position + avgHeight/2; j < c.peaks[i].position-avgHeight/2; j++ {
			valleyBlackCount := 0
			for l := 0; l < c.mat.Cols(); l++ {
				if c.mat.GetUCharAt(j, l) <= blackThreshold {
					valleyBlackCount++
				}
			}

			if minValue != 0 && valleyBlackCount <= minValue {
				minValue = valleyBlackCount
				minPosition = j
			}
		}

		v := &valleyStruct{
			id:         valleyIDCounter,
			chunkIndex: c.index,
			position:   minPosition,
		}
		valleyIDCounter++
		c.valleyIDs = append(c.valleyIDs, v.id)
		idToValley[v.id] = v
	}
	return idToValley
}

func (c *chunkStruct) calculateHistogram() {
	c.histogram = make([]int, c.mat.Rows())
	for i := 0; i < c.mat.Rows(); i++ {
		for j := 0; j < c.mat.Cols(); j++ {
			if c.mat.GetUCharAt(i, j) <= blackThreshold {
				c.histogram[i]++
			}
		}
	}
}

type valleyStruct struct {
	id         valleyID
	chunkIndex int
	position   int
	used       bool
}

type peakStruct struct {
	position int
	value    int
}
