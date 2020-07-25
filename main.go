package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

func main() {
	// open display window
	window := gocv.NewWindow("HWR")
	defer window.Close()

	// prepare image matrix
	origImg := gocv.IMRead("/home/david/Dropbox/Journal/2019-07-25.jpg", gocv.IMReadGrayScale)
	defer origImg.Close()
	if origImg.Empty() {
		panic("didn't load image")
	}
	//images := detectWords(origImg)
	// images := segmentLines(origImg)
	images := detectLines(origImg)

	finalImages := []gocv.Mat{}
	for _, i := range images {
		finalImages = append(finalImages, detectWords(i)[0])
	}
	images = finalImages

	i := 0
	for {
		toDraw := images[i%len(images)]
		window.IMShow(toDraw)
		if window.WaitKey(1000) == 27 {
			break
		}
		i++
	}
	for _, i := range images {
		_ = i.Close()
	}
}

func detectWords(origImg gocv.Mat) []gocv.Mat {
	blurred := gocv.NewMat()
	gocv.Blur(origImg, &blurred, image.Point{X: 5, Y: 5})

	thresholded := gocv.NewMat()
	gocv.AdaptiveThreshold(blurred, &thresholded, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 11)

	dilated := gocv.NewMat()
	dilationKernal := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 25, Y: 3})
	gocv.Dilate(thresholded, &dilated, dilationKernal)

	finalBinaryImage := dilated

	contours := gocv.FindContours(finalBinaryImage, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	blue := color.RGBA{0, 0, 255, 0}
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

	final := gocv.NewMat()

	gocv.CopyMakeBorder(origImg, &final, 0, 0, 0, 0, gocv.BorderConstant, blue)
	for _, r := range rectangles {
		gocv.Rectangle(&final, r, blue, 3)
	}

	return []gocv.Mat{final}
}

func detectLines(origImg gocv.Mat) []gocv.Mat {
	toOutput := []gocv.Mat{}

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
	bigChunk, _ := generateChunks(finalBinaryImage)

	// Get the initial lines
	idToValleys := make(map[valleyID]*valleyStruct, 50)
	bigChunk.findPeaksAndValleys(idToValleys)
	positions := make([]int, 0, len(bigChunk.valleyIDs))
	for _, valleyID := range bigChunk.valleyIDs {
		v := idToValleys[valleyID]

		y := v.position
		positions = append(positions, y)

		//pt1 := image.Pt(0, y)
		//pt2 := image.Pt(origImg.Cols(), y)
		//gocv.Line(&toOutput[0], pt1, pt2, color.RGBA{0, 255, 0, 50}, 3)
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

	fmt.Println(positions)
	fmt.Println(avgHeight)
	if minPosition > avgHeight {
		y := minPosition - avgHeight
		//pt1 := image.Pt(0, y)
		//pt2 := image.Pt(origImg.Cols(), y)
		//gocv.Line(&toOutput[0], pt1, pt2, color.RGBA{0, 255, 0, 50}, 3)
		positions = append([]int{y}, positions...)
	}
	if maxPosition+avgHeight < origImg.Rows() {
		y := maxPosition + avgHeight
		//pt1 := image.Pt(0, y)
		//pt2 := image.Pt(origImg.Cols(), y)
		//gocv.Line(&toOutput[0], pt1, pt2, color.RGBA{0, 255, 0, 50}, 3)
		positions = append(positions, y)
	}

	start := 0
	if positions[0]-avgHeight > 0 {
		start = positions[0] - avgHeight
	}
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
		start = end
		toOutput = append(toOutput, dest)
	}
	_ = finalBinaryImage.Close()

	// TODO:
	// - Chop up each line by each word with some more aggressive blur + contour detection
	// - Pipe each word into a ML model: https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5 / https://github.com/githubharald/SimpleHTR

	return toOutput
}

const (
	totalChunks    = 8
	bigChunkChunks = 8
	valleyFactor   = .60
	// I think this should be < width / totalChunks
	minAvgHeight   = 60
	blackThreshold = 0
)

type valleyID int

var (
	valleyIDCounter = valleyID(0)

	notPrimesArray []bool
	primes         []int
)

func init() {
	/*
		primes := make([]int, 0, 100)
		notPrimesArray = make([]bool, 10000)
		notPrimesArray[0] = true
		notPrimesArray[1] = true
		for i := 2; i < len(notPrimesArray); i++ {
			if notPrimesArray[i] {
				continue
			}
			primes = append(primes, i)
			for j := i * 2; j < len(notPrimesArray); j += i {
				notPrimesArray[j] = true
			}
		}*/
}

func addPrimesToVector(n int, probPrimes []int) {
	for i := 0; i < len(primes); i++ {
		for n%primes[i] != 0 {
			n /= primes[i]
			probPrimes[i]++
		}
	}
}

// Heavily inspired by: https://github.com/arthurflor23/text-segmentation/blob/master/src/imgproc/cpp/LineSegmentation.cpp
func segmentLines(origImage gocv.Mat) []gocv.Mat {
	toOutput := []gocv.Mat{}
	//
	// Steps:
	// Make it black or white
	blurred := gocv.NewMat()
	gocv.Blur(origImage, &blurred, image.Point{X: 5, Y: 5})

	thresholded := gocv.NewMat()
	gocv.AdaptiveThreshold(blurred, &thresholded, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 11)

	dilated := gocv.NewMat()
	dilationKernal := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 9, Y: 1})
	gocv.Dilate(thresholded, &dilated, dilationKernal)

	invertedDilated := gocv.NewMat()
	gocv.BitwiseNot(dilated, &invertedDilated)

	finalBinaryImage := invertedDilated
	contours := getContours(finalBinaryImage)
	toOutput = append(toOutput, origImage.Clone())
	blue := color.RGBA{0, 0, 255, 255}
	for _, c := range contours {
		gocv.Rectangle(&toOutput[len(toOutput)-1], c, blue, 2)
	}
	return toOutput

	// Generate chunks
	bigChunk, allChunks := generateChunks(finalBinaryImage)

	// Get the initial lines
	idToValleys := make(map[valleyID]*valleyStruct, 50)
	predictedLineHeight, lines := getInitialLines(finalBinaryImage, bigChunk, allChunks, idToValleys)

	//return []gocv.Mat{allChunks[len(allChunks)-1].mat, bigChunk.mat}

	toOutput = append(toOutput, origImage.Clone())
	renderLines(toOutput[len(toOutput)-1], lines)

	if len(lines) > 0 {
		// Generate regions
		_, averageLineHeight := generateRegions(finalBinaryImage, lines, predictedLineHeight)

		// Repair Lines
		repairLines(finalBinaryImage, lines, averageLineHeight, contours)

		// Generate regions (2)
		//regions, _ = generateRegions(thresholded, lines, predictedLineHeight)

		/*
			for _, r := range regions {
				toOutput = append(toOutput, r.region)
			}
		*/

		// Render lines
		toOutput = append(toOutput, origImage.Clone())
		renderLines(toOutput[len(toOutput)-1], lines)

		// Get regions
	}

	return toOutput[len(toOutput)-3:]
}

func getContours(binaryImage gocv.Mat) []image.Rectangle {
	contours := gocv.FindContours(binaryImage, gocv.RetrievalList, gocv.ChainApproxNone)

	approxCountours := make([][]image.Point, len(contours))
	boundingRectangles := make([]image.Rectangle, len(contours)-1)
	for i := 0; i < len(contours)-1; i++ {
		approxCountours[i] = gocv.ApproxPolyDP(contours[i], 1.0, true)
		boundingRectangles[i] = gocv.BoundingRect(approxCountours[i])
	}

	var r3 image.Rectangle
	var mergedRectangles []image.Rectangle

	for i, r1 := range boundingRectangles {
		isRepeated := false
		a1 := r1.Size().X * r1.Size().Y
		for j := i + 1; j < len(boundingRectangles); j++ {
			r2 := boundingRectangles[j]
			a2 := r2.Size().X * r2.Size().Y
			r3 = r1.Intersect(boundingRectangles[j])
			a3 := r3.Size().X * r3.Size().Y
			if a3 == a1 || a3 == a2 {
				isRepeated = true
				r3 = r1.Union(r2)

				// Why is this -2?
				if j == len(boundingRectangles)-2 {
					mergedRectangles = append(mergedRectangles, r3)
				}
				boundingRectangles[j] = r3
			}
		}
		if !isRepeated {
			mergedRectangles = append(mergedRectangles, r1)
		}
	}
	return mergedRectangles
}

type chunkStruct struct {
	index      int
	startPixel int
	chunkWidth int
	mat        gocv.Mat
	histogram  []int
	peaks      []peakStruct
	valleyIDs  []valleyID

	avgHeight, avgWhiteHeight int
	linesCount                int
}

func (c *chunkStruct) findPeaksAndValleys(idToValley map[valleyID]*valleyStruct) {
	c.calculateHistogram()

	for i := 1; i < len(c.histogram)-1; i++ {
		leftVal := c.histogram[i-1]
		centerVal := c.histogram[i]
		rightVal := c.histogram[i+1]

		// See if we're looking at a peak
		if centerVal >= leftVal && centerVal >= rightVal {
			if len(c.peaks) > 0 {
				last := c.peaks[len(c.peaks)-1]
				if (i-last.position) <= c.avgHeight/2 && centerVal >= last.value {
					c.peaks[len(c.peaks)-1].position = i
					c.peaks[len(c.peaks)-1].value = centerVal
				} else if (i-last.position) <= c.avgHeight/2 && centerVal < last.value {
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

	for i := 1; i < len(c.peaks); i++ {
		minPosition := (c.peaks[i-1].position + c.peaks[i].position) / 2
		minValue := c.histogram[minPosition]

		for j := c.peaks[i-1].position + c.avgHeight/2; j < c.peaks[i].position-c.avgHeight-minAvgHeight; j++ {
			valleyBlackCount := 0
			for l := 0; l < c.mat.Cols(); l++ {
				if c.mat.GetUCharAt(j, l) == 0 {
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
}

func (c *chunkStruct) calculateHistogram() {
	c.histogram = make([]int, c.mat.Rows())
	c.linesCount, c.avgHeight, c.avgWhiteHeight = 0, 0, 0

	blackCount, currentHeight, currentWhiteCount, whiteLinesCount := 0, 0, 0, 0
	var whiteSpaces []int

	for i := 0; i < c.mat.Rows(); i++ {
		blackCount = 0
		for j := 0; j < c.mat.Cols(); j++ {
			if c.mat.GetUCharAt(i, j) <= blackThreshold {
				blackCount++
				c.histogram[i]++
			}
		}
		if blackCount > 0 {
			currentHeight++
			if currentWhiteCount > 0 {
				whiteSpaces = append(whiteSpaces, currentWhiteCount)
				currentWhiteCount = 0
			}
		} else {
			currentWhiteCount++
			if currentHeight > 0 {
				c.linesCount++
				c.avgHeight += currentHeight
				currentHeight = 0
			}
		}
	}

	sort.Slice(whiteSpaces, func(i, j int) bool {
		return whiteSpaces[i] < whiteSpaces[j]
	})
	for _, space := range whiteSpaces {
		if space > int(valleyFactor*float64(c.avgHeight)) {
			break
		}
		c.avgWhiteHeight += space
		whiteLinesCount++
	}
	if whiteLinesCount > 0 {
		c.avgWhiteHeight /= whiteLinesCount
	}
	if c.linesCount > 0 {
		c.avgHeight /= c.linesCount
	}

	c.avgHeight += int(float64(c.avgHeight) / 2.0)
	if c.avgHeight < minAvgHeight {
		c.avgHeight = minAvgHeight
	}
}

type valleyStruct struct {
	id         valleyID
	chunkIndex int
	position   int
	used       bool
	line       *lineStruct
}

func (v *valleyStruct) isSame(v2 *valleyStruct) bool {
	return v.id == v2.id
}

type peakStruct struct {
	position int
	value    int
}

type lineStruct struct {
	above, below                   *regionStruct
	valleyIDs                      []valleyID
	minRowPosition, maxRowPosition int
	points                         []image.Point
}

func (l *lineStruct) generateInitialPoints(
	chunkWidth int,
	imageWidth int,
	idToValley map[valleyID]*valleyStruct,
) {
	c, previousRow := 0, 0
	sort.Slice(l.valleyIDs, func(i, j int) bool {
		return l.valleyIDs[i] < l.valleyIDs[j]
	})

	firstV := idToValley[l.valleyIDs[0]]
	if firstV.chunkIndex == totalChunks {
		panic("First valley was a big chunk")
	}
	if firstV.chunkIndex > 0 {
		previousRow = firstV.position
		l.maxRowPosition = previousRow
		l.minRowPosition = previousRow

		for j := 0; j < firstV.chunkIndex*chunkWidth; j++ {
			if c == j {
				l.points = append(l.points, image.Point{X: previousRow, Y: j})
			}
			c++
		}
	}

	for _, vid := range l.valleyIDs {
		v := idToValley[vid]

		chunkIndex := v.chunkIndex
		chunkRow := v.position
		if chunkRow < l.minRowPosition {
			l.minRowPosition = chunkRow
		}
		if chunkRow > l.maxRowPosition {
			l.maxRowPosition = chunkRow
		}

		chunkStartColumn := chunkIndex * chunkWidth
		for j := chunkStartColumn; j < chunkStartColumn+chunkWidth && j < imageWidth; j++ {
			if c == j {
				l.points = append(l.points, image.Point{X: chunkRow, Y: j})
			}
			c++
		}
		if previousRow != chunkRow {
			previousRow = chunkRow
		}
	}

	lastValley := idToValley[l.valleyIDs[len(l.valleyIDs)-1]]
	chunkIndex := lastValley.chunkIndex
	chunkRow := lastValley.position
	for j := chunkIndex*chunkWidth + chunkWidth; j < imageWidth; j++ {
		if c == j {
			l.points = append(l.points, image.Point{X: chunkRow, Y: j})
		}
		c++
	}
}

type regionStruct struct {
	regionIndex int
	region      gocv.Mat
	top, bottom *lineStruct
	height      int
	rowOffset   int
	covariance  gocv.Mat
	mean        []float32
}

// Returns true if the region is all zeros, false otherwise
func (r *regionStruct) updateRegion(img gocv.Mat, regionIndex int) bool {
	r.regionIndex = regionIndex
	minRegionRow := 0
	if r.top != nil {
		minRegionRow = r.top.minRowPosition
	}
	r.rowOffset = minRegionRow
	maxRegionRow := img.Rows()
	if r.bottom != nil {
		maxRegionRow = r.bottom.maxRowPosition
	}

	start := minRegionRow
	if maxRegionRow < start {
		start = maxRegionRow
	}
	end := maxRegionRow
	if minRegionRow > end {
		end = minRegionRow
	}

	regionTotalRows := end - start
	r.region = gocv.NewMatWithSize(regionTotalRows, img.Cols(), gocv.MatTypeCV8U)
	nonzero := 0
	for c := 0; c < img.Cols(); c++ {
		start := 0
		if r.top != nil {
			start = r.top.points[c].X
		}
		end := img.Rows() - 1
		if r.bottom != nil {
			end = r.bottom.points[c].X
		}
		if end-start > r.height {
			r.height = end - start
		}
		for row := 0; row < regionTotalRows; row++ {
			origRow := row + minRegionRow
			val := uint8(255)
			if origRow >= start && origRow < end {
				val = img.GetUCharAt(origRow, c)
			}
			if val != 0 {
				nonzero++
			}
			r.region.SetUCharAt(row, c, val)
		}
	}
	r.calculateMean()
	r.calculateCovariance()
	return nonzero == r.region.Cols()*r.region.Rows()
}

func (r *regionStruct) calculateMean() {
	mean := make([]float32, 2)
	n := 0
	for i := 0; i < r.region.Rows(); i++ {
		for j := 0; j < r.region.Cols(); j++ {
			if r.region.GetUCharAt(i, j) == 255 {
				continue
			}

			if n == 0 {
				mean[0] = float32(i + r.rowOffset)
				mean[1] = float32(j)
			} else {
				fn := float32(n)
				f := (fn - 1.0) / fn
				mean[0] = f*mean[0] + (1-f)*float32(i+r.rowOffset)
				mean[1] = f*mean[1] + (1-f)*float32(j)
			}
			n++
		}
	}
	r.mean = mean
}

func (r *regionStruct) calculateCovariance() {
	r.covariance = gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F)
	n := 0
	var sumISquared, sumJSquared, sumIJ float32 = 0.0, 0.0, 0.0
	for i := 0; i < r.region.Rows(); i++ {
		for j := 0; j < r.region.Cols(); j++ {
			if r.region.GetUCharAt(i, j) == 255 {
				continue
			}

			newI := float32(i+r.rowOffset) - r.mean[0]
			newJ := float32(j) - r.mean[1]

			sumISquared += newI * newI
			sumIJ += newI * newJ
			sumJSquared += newJ * newJ
			n++
		}
	}

	if n == 0 {
		return
	}
	nf := float32(n)
	r.covariance.SetFloatAt(0, 0, sumISquared/nf)
	r.covariance.SetFloatAt(0, 1, sumIJ/nf)
	r.covariance.SetFloatAt(1, 0, sumIJ/nf)
	r.covariance.SetFloatAt(1, 1, sumJSquared/nf)
}

func (r *regionStruct) biVariateGaussianDensity(point gocv.Mat) float64 {
	point.SetFloatAt(0, 0, point.GetFloatAt(0, 0)-r.mean[0])
	point.SetFloatAt(0, 1, point.GetFloatAt(0, 1)-r.mean[1])

	pointTranspose := gocv.NewMat()
	gocv.Transpose(point, &pointTranspose)

	invCovariance := gocv.NewMat()
	gocv.Invert(r.covariance, &invCovariance, 0)

	dst := point.MultiplyMatrix(invCovariance)
	dst2 := dst.MultiplyMatrix(pointTranspose)

	v1 := float64(dst2.GetFloatAt(0, 0))

	c := r.covariance.Clone()
	c.MultiplyFloat(2 * math.Pi)
	det := gocv.Determinant(c)
	return v1 * math.Sqrt(det)
}

func generateChunks(origImage gocv.Mat) (*chunkStruct, []*chunkStruct) {
	width := origImage.Cols() / totalChunks

	chunks := make([]*chunkStruct, totalChunks)
	for i, startPixel := 0, 0; i < totalChunks; i++ {
		chunks[i] = &chunkStruct{
			index:      i,
			startPixel: startPixel,
			chunkWidth: width,
			mat:        origImage.Region(image.Rect(startPixel, 0, startPixel+width, origImage.Rows())),
		}
		startPixel += width
	}

	bigWidth := bigChunkChunks * width
	bigOffset := origImage.Cols() - bigWidth
	bigChunk := &chunkStruct{
		index:      totalChunks,
		startPixel: bigOffset,
		chunkWidth: bigWidth,
		mat:        origImage.Region(image.Rect(bigOffset, 0, bigOffset+bigWidth, origImage.Rows())),
	}
	return bigChunk, chunks
}

// Returns predicted line height and the initial lines
func getInitialLines(
	origImage gocv.Mat,
	bigChunk *chunkStruct,
	chunks []*chunkStruct,
	idToValleys map[valleyID]*valleyStruct,
) (int, []*lineStruct) {
	lines := make([]*lineStruct, 0, len(chunks))
	numberOfHeights, valleysMinAbsoluteDistance := 0, 0
	var normalChunkWidth int
	for i := 0; i <= len(chunks); i++ {
		var c *chunkStruct
		if i < len(chunks) {
			c = chunks[i]
			normalChunkWidth = c.chunkWidth
		} else {
			c = bigChunk
		}
		c.findPeaksAndValleys(idToValleys)
		if i < len(chunks) {
			if c.avgHeight > 0 {
				numberOfHeights++
			}
			valleysMinAbsoluteDistance += c.avgHeight
		}
	}
	valleysMinAbsoluteDistance /= numberOfHeights
	for i, vid := range bigChunk.valleyIDs {
		v := idToValleys[vid]
		if v.used {
			continue
		}
		v.used = true
		l := &lineStruct{
			valleyIDs: []valleyID{vid},
		}
		connectValleys(chunks, idToValleys, len(chunks)-1, vid, l, valleysMinAbsoluteDistance)
		if len(l.valleyIDs) > 1 {
			l.generateInitialPoints(normalChunkWidth, origImage.Cols(), idToValleys)
			lines = append(lines, l)
		} else {
			fmt.Println("Droppping line from valley", i)
		}
	}

	return valleysMinAbsoluteDistance, lines
}

func connectValleys(
	chunks []*chunkStruct,
	idToValleys map[valleyID]*valleyStruct,
	i int,
	current valleyID,
	l *lineStruct,
	valleysMinAbsoluteDistance int,
) {
	if i <= 0 || len(chunks[i].valleyIDs) == 0 {
		return
	}

	currentValley := idToValleys[current]
	connectedTo := -1
	minDistance := math.MaxInt64
	for j, vid := range chunks[i].valleyIDs {
		v := idToValleys[vid]
		if v.used {
			continue
		}

		dist := currentValley.position - v.position
		if dist < 0 {
			dist = -dist
		}
		if minDistance > dist && dist <= valleysMinAbsoluteDistance {
			minDistance = dist
			connectedTo = j
		}
	}
	if connectedTo == -1 {
		return
	}

	newV := idToValleys[chunks[i].valleyIDs[connectedTo]]
	l.valleyIDs = append(l.valleyIDs, newV.id)
	newV.used = true
	connectValleys(chunks, idToValleys, i-1, newV.id, l, valleysMinAbsoluteDistance)
}

func renderLines(mat gocv.Mat, lines []*lineStruct) {
	for _, l := range lines {
		lastRow := -1
		for _, p := range l.points {
			mat.SetUCharAt(p.X, p.Y, 255)
			if lastRow != -1 && p.X != lastRow {
				min := p.X
				if lastRow < min {
					min = lastRow
				}
				max := p.X
				if lastRow > max {
					max = lastRow
				}
				for i := min; i < max; i++ {
					mat.SetUCharAt(i, p.Y, 255)
				}
			}
			lastRow = p.X
		}
	}
}

func generateRegions(
	origImage gocv.Mat,
	lines []*lineStruct,
	predictedLineHeight int,
) ([]*regionStruct, int) {
	sort.Slice(lines, func(i, j int) bool {
		return lines[i].minRowPosition < lines[j].minRowPosition
	})

	lineRegions := make([]*regionStruct, 0, len(lines))
	r := &regionStruct{bottom: lines[0]}
	r.updateRegion(origImage, 0)

	lines[0].above = r
	lineRegions = append(lineRegions, r)

	regionMaxHeight := int(float64(predictedLineHeight) * 2.5)
	averageLineHeight := 0
	toExclude := 0
	if r.height < regionMaxHeight {
		averageLineHeight += r.height
	} else {
		toExclude++
	}

	for i := range lines {
		topLine := lines[i]
		var bottomLine *lineStruct
		if i < len(lines)-1 {
			bottomLine = lines[i+1]
		}

		r := &regionStruct{top: topLine, bottom: bottomLine}
		isTrivial := r.updateRegion(origImage, i)
		topLine.below = r
		if bottomLine != nil {
			bottomLine.above = r
		}
		if !isTrivial {
			lineRegions = append(lineRegions, r)
			if r.height < regionMaxHeight {
				averageLineHeight += r.height
			} else {
				toExclude++
			}
		}
	}
	if len(lineRegions)-toExclude > 0 {
		averageLineHeight /= len(lineRegions) - toExclude
	}
	return lineRegions, averageLineHeight
}

func repairLines(
	thresholdedImage gocv.Mat,
	lines []*lineStruct,
	averageLineHeight int,
	contours []image.Rectangle,
) {
	for _, l := range lines {
		columnProcessed := make(map[int]bool, len(l.points))
		for i := 0; i < len(l.points); i++ {
			p := l.points[i]
			x := p.X
			y := p.Y
			if thresholdedImage.GetUCharAt(p.X, p.Y) == 255 {
				if i == 0 {
					continue
				}
				blackFound := false

				if l.points[i-1].X != l.points[i].X {
					minRow := l.points[i].X
					maxRow := l.points[i].X
					if l.points[i-1].X < minRow {
						minRow = l.points[i-1].X
					}
					if l.points[i-1].X > maxRow {
						maxRow = l.points[i-1].X
					}

					for j := minRow; j <= maxRow && !blackFound; j++ {
						if thresholdedImage.GetUCharAt(j, l.points[i-1].Y) == 0 {
							x = j
							y = l.points[i-1].Y
							blackFound = true
						}
					}
				}
				if !blackFound {
					continue
				}
			}
			if columnProcessed[y] {
				continue
			}
			columnProcessed[y] = true

			for _, c := range contours {
				if y >= c.Min.X && y <= c.Max.X && x >= c.Min.Y && x <= c.Max.Y {
					// We want the contours to be smaller than a line because they should be within one.
					// TODO move this out
					if c.Max.Y-c.Min.Y > int(float64(averageLineHeight)*0.9) {
						continue
					}

					isComponentAbove := componentBelongsToAboveRegion(thresholdedImage, l, c)

					var newRow int
					if !isComponentAbove {
						newRow = c.Min.Y
						if newRow < l.minRowPosition {
							l.minRowPosition = newRow
						}
					} else {
						newRow = c.Max.Y
						if newRow > l.maxRowPosition {
							l.maxRowPosition = newRow
						}
					}
					for k := c.Min.X; k < c.Min.X+c.Size().X; k++ {
						l.points[k].X = newRow
					}
					i = c.Max.X
					break
				}
			}
		}
	}
}

func componentBelongsToAboveRegion(
	thresholdedImage gocv.Mat,
	line *lineStruct,
	contour image.Rectangle,
) bool {
	//probAbovePrimes := make([]int, len(primes))
	//probBelowPrimes := make([]int, len(primes))
	n := 0
	newProbAbove, newProbBelow := int64(0), int64(0)

	for i := contour.Min.X; i < contour.Min.X+contour.Size().X; i++ {
		for j := contour.Min.Y; j < contour.Min.Y+contour.Size().Y; j++ {
			if thresholdedImage.GetUCharAt(j, i) == 255 {
				continue
			}
			n++

			contourPoint := gocv.NewMatWithSize(1, 2, gocv.MatTypeCV32F)
			contourPoint.SetFloatAt(0, 0, float32(j))
			contourPoint.SetFloatAt(0, 1, float32(i))

			if line.above != nil {
				newProbAbove += int64(line.above.biVariateGaussianDensity(contourPoint.Clone()))
			}
			if line.below != nil {
				newProbBelow += int64(line.below.biVariateGaussianDensity(contourPoint.Clone()))
			}

			//addPrimesToVector(newProbAbove, probAbovePrimes)
			//addPrimesToVector(newProbBelow, probBelowPrimes)
		}
	}
	return newProbAbove < newProbBelow

	/*
		probAbove, probBelow := 0, 0
		for k := 0; k < len(probAbovePrimes); k++ {
			mini := probAbovePrimes[k]
			if probBelowPrimes[k] < mini {
				mini = probBelowPrimes[k]
			}
			probAbovePrimes[k] -= mini
			probBelowPrimes[k] -= mini

			probAbove += probAbovePrimes[k] * primes[k]
			probBelow += probBelowPrimes[k] * primes[k]
		}

		// Doesn't this seem backwards??
		return probAbove < probBelow
	*/
}
