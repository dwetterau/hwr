package main

import (
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

func main() {
	// open display window
	window := gocv.NewWindow("HRR")
	defer window.Close()

	// prepare image matrix
	origImg := gocv.IMRead("/home/david/Dropbox/Journal/2019-07-25.jpg", gocv.IMReadGrayScale)
	defer origImg.Close()
	if origImg.Empty() {
		panic("didn't load image")
	}
	blurred := gocv.NewMat()
	defer blurred.Close()
	gocv.Blur(origImg, &blurred, image.Point{X: 5, Y: 5})

	thresholded := gocv.NewMat()
	defer thresholded.Close()

	gocv.AdaptiveThreshold(blurred, &thresholded, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 15)

	dilated := gocv.NewMat()
	defer dilated.Close()
	dilationKernal := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 6, Y: 1})
	dilationKernalTall := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 5, Y: 3})
	src, dst := &thresholded, &dilated

	newSrc := gocv.NewMat()
	defer newSrc.Close()

	iterations := 5
	for i := 0; i < iterations; i++ {
		k := dilationKernal
		if i == iterations-1 {
			k = dilationKernalTall
		}
		gocv.Dilate(*src, dst, k)
		newDst := &newSrc
		if i != 0 {
			newDst = src
		}
		src = dst
		dst = newDst
	}
	dilated = *src

	contours := gocv.FindContours(dilated, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	blue := color.RGBA{0, 0, 255, 0}
	rectangles := make([]image.Rectangle, 0, len(contours))
	for _, c := range contours {
		r := gocv.BoundingRect(c)
		s := r.Size()
		if s.X*s.Y > 1000 {
			rectangles = append(rectangles, r)
		}
	}
	images := []gocv.Mat{origImg, blurred, thresholded, dilated}
	i := 0
	for {
		toDraw := images[i%len(images)]
		for _, r := range rectangles {
			gocv.Rectangle(&toDraw, r, blue, 3)
		}
		window.IMShow(toDraw)
		if window.WaitKey(1000) == 27 {
			break
		}
		i++
	}
}
