#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;
using namespace std;

// Simple histogram renderer (values -> bar plot as an image)
Mat plotHistogram(const vector<float>& vals, int bins = 30, Size canvas = Size(640, 400)) {
    Mat plot(canvas, CV_8UC3, Scalar(255, 255, 255));
    if (vals.empty() || bins < 1) return plot;

    auto [minIt, maxIt] = minmax_element(vals.begin(), vals.end());
    float vmin = *minIt, vmax = *maxIt;
    if (vmin == vmax) vmax = vmin + 1e-6f; // avoid zero range

    vector<int> counts(bins, 0);
    for (float v : vals) {
        int idx = static_cast<int>((v - vmin) / (vmax - vmin) * (bins - 1));
        counts[idx]++;
    }
    int maxCount = *max_element(counts.begin(), counts.end());
    if (maxCount == 0) return plot;

    int left = 50, right = 20, bottom = 40, top = 20;
    int W = canvas.width - left - right;
    int H = canvas.height - top - bottom;

    // Axes
    line(plot, Point(left, canvas.height - bottom), Point(canvas.width - right, canvas.height - bottom), Scalar(0, 0, 0), 1);
    line(plot, Point(left, canvas.height - bottom), Point(left, top), Scalar(0, 0, 0), 1);

    // Bars
    double barW = static_cast<double>(W) / bins;
    for (int i = 0; i < bins; ++i) {
        int h = static_cast<int>((double)counts[i] / maxCount * H);
        int x0 = left + static_cast<int>(i * barW);
        int x1 = left + static_cast<int>((i + 1) * barW) - 1;
        int y0 = canvas.height - bottom;
        rectangle(plot, Point(x0, y0 - h), Point(x1, y0 - 1), Scalar(100, 100, 255), FILLED);
    }

    // Tick labels (min/max)
    putText(plot, format("%.2f", vmin), Point(left, canvas.height - 10),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(plot, format("%.2f", vmax), Point(canvas.width - right - 60, canvas.height - 10),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(plot, "count", Point(5, top + 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(plot, "distance", Point(canvas.width / 2 - 40, canvas.height - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    return plot;
}

// ----------------- Robust panorama utilities -----------------
static void computeWarpedCanvas(const Mat& img1, const Mat& img2, const Mat& H12,
    Size& canvasSize, Mat& T21 /* translation */) {
    vector<Point2f> c1 = { {0,0}, {float(img1.cols - 1),0},
                           {float(img1.cols - 1), float(img1.rows - 1)}, {0, float(img1.rows - 1)} };
    vector<Point2f> c1_w;
    perspectiveTransform(c1, c1_w, H12);

    vector<Point2f> c2 = { {0,0}, {float(img2.cols - 1),0},
                           {float(img2.cols - 1), float(img2.rows - 1)}, {0, float(img2.rows - 1)} };

    float minx = numeric_limits<float>::max(), miny = numeric_limits<float>::max();
    float maxx = -numeric_limits<float>::max(), maxy = -numeric_limits<float>::max();
    auto upd = [&](const Point2f& p) { minx = min(minx, p.x); miny = min(miny, p.y);
    maxx = max(maxx, p.x); maxy = max(maxy, p.y); };
    for (auto& p : c1_w) upd(p);
    for (auto& p : c2)   upd(p);

    float tx = (minx < 0) ? -minx : 0.f;
    float ty = (miny < 0) ? -miny : 0.f;

    int W = int(ceil(maxx + tx));
    int H = int(ceil(maxy + ty));
    canvasSize = Size(max(W, 1), max(H, 1));

    T21 = (Mat_<double>(3, 3) << 1, 0, tx, 0, 1, ty, 0, 0, 1);
}

static Mat blendOverlay(const Mat& warped1, const Mat& img2_on_canvas,
    const Mat& mask1_255, const Mat& /*mask2_255*/) {
    Mat pano = img2_on_canvas.clone();
    warped1.copyTo(pano, mask1_255);
    return pano;
}

// Feathering with proper binary masks for distanceTransform
static Mat blendFeather(const Mat& warped1, const Mat& img2_on_canvas,
    const Mat& mask1_255, const Mat& mask2_255) {
    // Convert 255 masks to 0/1 (CV_8U) for distanceTransform
    Mat m1_01, m2_01;
    threshold(mask1_255, m1_01, 0, 1, THRESH_BINARY);
    threshold(mask2_255, m2_01, 0, 1, THRESH_BINARY);

    // Overlap check
    Mat overlap01;
    bitwise_and(m1_01, m2_01, overlap01);
    if (countNonZero(overlap01) == 0) {
        // No overlap → overlay is fine
        return blendOverlay(warped1, img2_on_canvas, mask1_255, mask2_255);
    }

    // Distance to nearest background for foreground pixels (L2, 3x3)
    Mat d1, d2;
    distanceTransform(m1_01, d1, DIST_L2, 3);
    distanceTransform(m2_01, d2, DIST_L2, 3);

    // Convert images to float
    Mat warped1f, img2f;
    warped1.convertTo(warped1f, CV_32F);
    img2_on_canvas.convertTo(img2f, CV_32F);

    // Convert distances to float, compute weights
    Mat d1f, d2f; d1.convertTo(d1f, CV_32F); d2.convertTo(d2f, CV_32F);

    // Normalize weights only in union region to avoid undefined areas
    Mat union01; bitwise_or(m1_01, m2_01, union01);
    Mat denom = d1f + d2f;
    denom.setTo(1.0f, denom <= 1e-6f); // avoid div-by-zero

    Mat w1 = d1f / denom;
    Mat w2 = d2f / denom;

    // Ensure exclusive regions are fully weighted to their image
    // where m1=1 & m2=0 → w1=1, w2=0
    // where m2=1 & m1=0 → w2=1, w1=0
    w1.setTo(1.0f, (m1_01 == 1) & (m2_01 == 0));
    w2.setTo(0.0f, (m1_01 == 1) & (m2_01 == 0));
    w2.setTo(1.0f, (m2_01 == 1) & (m1_01 == 0));
    w1.setTo(0.0f, (m2_01 == 1) & (m1_01 == 0));

    // Outside the union → weights 0
    w1.setTo(0.0f, union01 == 0);
    w2.setTo(0.0f, union01 == 0);

    // Expand weights to 3 channels
    Mat w1c, w2c; {
        vector<Mat> w1v(3, w1), w2v(3, w2);
        merge(w1v, w1c); merge(w2v, w2c);
    }

    Mat pano32 = warped1f.mul(w1c) + img2f.mul(w2c);
    Mat pano; pano32.convertTo(pano, CV_8U);
    return pano;
}

static void buildPanoramas(const Mat& img1_color, const Mat& img2_color, const Mat& H12,
    const string& outOverlay, const string& outFeather) {
    Size canvasSize; Mat T;
    computeWarpedCanvas(img1_color, img2_color, H12, canvasSize, T);

    // Warp img1 into canvas
    Mat H_canvas = T * H12;
    Mat warped1;
    warpPerspective(img1_color, warped1, H_canvas, canvasSize);

    // Place img2 on canvas via pure translation T
    Mat img2_on_canvas;
    warpPerspective(img2_color, img2_on_canvas, T, canvasSize);

    // Build binary masks (255 where valid, 0 elsewhere)
    Mat gray1, gray2, mask1_255, mask2_255;
    cvtColor(warped1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2_on_canvas, gray2, COLOR_BGR2GRAY);
    mask1_255 = Mat::zeros(canvasSize, CV_8U);
    mask2_255 = Mat::zeros(canvasSize, CV_8U);
    mask1_255.setTo(255, gray1 > 0);
    mask2_255.setTo(255, gray2 > 0);

    // Build panoramas
    Mat pano_overlay = blendOverlay(warped1, img2_on_canvas, mask1_255, mask2_255);
    imwrite(outOverlay, pano_overlay);

    Mat pano_feather = blendFeather(warped1, img2_on_canvas, mask1_255, mask2_255);
    imwrite(outFeather, pano_feather);
}

size_t matchAndSaveSIFT_BF(const string& img1Path, const string& img2Path, const string& outPath, float ratio = 0.75f) {
    Mat img1 = imread(img1Path, IMREAD_GRAYSCALE);
    Mat img2 = imread(img2Path, IMREAD_GRAYSCALE);
    Mat img1c = imread(img1Path, IMREAD_COLOR);
    Mat img2c = imread(img2Path, IMREAD_COLOR);

    if (img1.empty() || img2.empty() || img1c.empty() || img2c.empty()) {
        cerr << "Failed to read input images.\n";
        return 0;
    }

    resize(img1, img1, Size(480, 480));
    resize(img2, img2, Size(480, 480));
    resize(img1c, img1c, Size(480, 480));
    resize(img2c, img2c, Size(480, 480));

    Ptr<SIFT> sift = SIFT::create();

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;
    sift->detectAndCompute(img1, noArray(), kp1, des1);
    sift->detectAndCompute(img2, noArray(), kp2, des2);

    Mat img1Keypoints, img2Keypoints;
    drawKeypoints(img1, kp1, img1Keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2, kp2, img2Keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    putText(img1Keypoints, "Keypoints: " + to_string(kp1.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 4);
    putText(img1Keypoints, "Keypoints: " + to_string(kp1.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    putText(img2Keypoints, "Keypoints: " + to_string(kp2.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 4);
    putText(img2Keypoints, "Keypoints: " + to_string(kp2.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    BFMatcher bf(NORM_L2);

    // Measure descriptor matching time (knnMatch)
    TickMeter tm;
    tm.start();
    vector<vector<DMatch>> knn;
    bf.knnMatch(des1, des2, knn, 2);
    tm.stop();
    double match_ms = tm.getTimeMilli();
    cout << "Descriptor matching time (BF knnMatch): " << match_ms << " ms\n";


    vector<DMatch> good;
    vector<float> dist_all;   // first-NN distances from knn
    dist_all.reserve(knn.size());

    for (auto& m : knn) {
        if (m.size() == 2) {
            dist_all.push_back(m[0].distance);
            if (m[0].distance < ratio * m[1].distance)
                good.push_back(m[0]);
        }
    }

    // Distances of the “good” matches
    vector<float> dist_good;
    dist_good.reserve(good.size());
    for (const auto& d : good) dist_good.push_back(d.distance);

    // --- RANSAC Homography from good matches ---
    vector<DMatch> inliers; // inlier matches for visualization/return
    Mat H;                  // homography

    if (good.size() < 4) {
        cerr << "Not enough matches for homography (need >= 4). Good matches: " << good.size() << "\n";
    }
    else {
        // Collect corresponding points
        vector<Point2f> pts1, pts2;
        pts1.reserve(good.size());
        pts2.reserve(good.size());
        for (const auto& m : good) {
            pts1.push_back(kp1[m.queryIdx].pt); // img1 -> query
            pts2.push_back(kp2[m.trainIdx].pt); // img2 -> train
        }

        // RANSAC homography
        // Threshold ~3.0px is reasonable at 480x480; tune as needed
        Mat inlierMask;
        H = findHomography(pts1, pts2, RANSAC, 3.0, inlierMask, 2000, 0.995);

        if (H.empty()) {
            cerr << "Homography estimation failed.\n";
        }
        else {
            inliers.reserve(good.size());
            for (size_t i = 0; i < good.size(); ++i) {
                if (inlierMask.at<uchar>(static_cast<int>(i))) {
                    inliers.push_back(good[i]);
                }
            }
            cout << "RANSAC inliers: " << inliers.size() << " / " << good.size() << "\n";
        }
    }

    // Prefer drawing inliers if available
    const vector<DMatch>& matchesToDraw = (!inliers.empty() ? inliers : good);
    string legend = (!inliers.empty()
        ? ("Inlier matches (RANSAC): " + to_string(inliers.size()))
        : ("Good matches (no/failed RANSAC): " + to_string(good.size())));


    Mat vis;
    drawMatches(img1, kp1, img2, kp2, matchesToDraw, vis,
        Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    putText(vis, legend, Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    // Histograms
    Mat histAll = plotHistogram(dist_all, 30, Size(640, 400));
    Mat histGood = plotHistogram(dist_good, 30, Size(640, 400));
    putText(histAll, "All first-NN distances (SIFT/L2)", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    putText(histGood, "Good match distances (SIFT/L2)", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);

    // Show base visuals
    imshow("Image 1 (resized, with keypoints) [SIFT]", img1Keypoints);
    imshow("Image 2 (resized, with keypoints) [SIFT]", img2Keypoints);
    imshow("SIFT Matches", vis);
    imshow("Histogram - All distances", histAll);
    imshow("Histogram - Good distances", histGood);

    // Panorama stitching (only if homography succeeded)
    if (!H.empty() && !inliers.empty()) {
        buildPanoramas(img1c, img2c, H, "panorama_overlay.jpg", "panorama_feather.jpg");
        cout << "Saved panorama_overlay.jpg and panorama_feather.jpg\n";

        // Display panoramas
        Mat pano_overlay = imread("panorama_overlay.jpg");
        Mat pano_feather = imread("panorama_feather.jpg");
        if (!pano_overlay.empty()) imshow("Panorama - Overlay", pano_overlay);
        if (!pano_feather.empty()) imshow("Panorama - Feathering", pano_feather);
    }
    else {
        cerr << "Skipping panorama stitching (no homography).\n";
    }



    waitKey(0);

    // Save outputs
    imwrite(outPath, vis);
    imwrite("hist_all.jpg", histAll);
    imwrite("hist_good.jpg", histGood);

    return (!inliers.empty() ? inliers.size() : good.size()); 
}


size_t matchAndSaveORB_BF(const string& img1Path, const string& img2Path, const string& outPath, float ratio = 0.75f) {
    Mat img1 = imread(img1Path, IMREAD_GRAYSCALE);
    Mat img2 = imread(img2Path, IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "Failed to read input images.\n";
        return 0;
    }

    resize(img1, img1, Size(480, 480));
    resize(img2, img2, Size(480, 480));

    Ptr<ORB> orb = ORB::create();

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;
    orb->detectAndCompute(img1, noArray(), kp1, des1);
    orb->detectAndCompute(img2, noArray(), kp2, des2);

    // Visualize keypoints
    Mat img1Keypoints, img2Keypoints;
    drawKeypoints(img1, kp1, img1Keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2, kp2, img2Keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    putText(img1Keypoints, "Keypoints: " + to_string(kp1.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 4);
    putText(img1Keypoints, "Keypoints: " + to_string(kp1.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    putText(img2Keypoints, "Keypoints: " + to_string(kp2.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 4);
    putText(img2Keypoints, "Keypoints: " + to_string(kp2.size()), Point(20, 40),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    if (des1.empty() || des2.empty()) {
        cerr << "No descriptors found.\n";
        imshow("Image 1 (resized, with keypoints)", img1Keypoints);
        imshow("Image 2 (resized, with keypoints)", img2Keypoints);
        waitKey(0);
        return 0;
    }

    // ORB uses Hamming distance
    BFMatcher bf(NORM_HAMMING);

    // Time the descriptor matching
    TickMeter tm;
    tm.start();
    vector<vector<DMatch>> knn;
    bf.knnMatch(des1, des2, knn, 2);
    tm.stop();
    double match_ms = tm.getTimeMilli();
    cout << "Descriptor matching time (BF knnMatch, ORB/Hamming): " << match_ms << " ms\n";

    // Lowe's ratio test and distance collection
    vector<DMatch> good;
    vector<float> dist_all;
    dist_all.reserve(knn.size());

    for (auto& m : knn) {
        if (m.size() == 2) {
            dist_all.push_back(m[0].distance);
            if (m[0].distance < ratio * m[1].distance)
                good.push_back(m[0]);
        }
    }

    vector<float> dist_good;
    dist_good.reserve(good.size());
    for (const auto& d : good) dist_good.push_back(d.distance);

    // Draw matches
    Mat vis;
    drawMatches(img1, kp1, img2, kp2, good, vis,
        Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    putText(vis, "Good matches: " + to_string(good.size()),
        Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    // Histograms (reuse your plotHistogram)
    Mat histAll = plotHistogram(dist_all, 30, Size(640, 400));
    Mat histGood = plotHistogram(dist_good, 30, Size(640, 400));
    putText(histAll, "All first-NN distances (ORB/Hamming)", Point(10, 25),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    putText(histGood, "Good match distances (ORB/Hamming)", Point(10, 25),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);

    // Show
    imshow("Image 1 (resized, with keypoints) [ORB]", img1Keypoints);
    imshow("Image 2 (resized, with keypoints) [ORB]", img2Keypoints);
    imshow("ORB Matches", vis);
    imshow("Histogram - All distances [ORB]", histAll);
    imshow("Histogram - Good distances [ORB]", histGood);
    waitKey(0);

    // Save
    imwrite(outPath, vis);
    imwrite("orb_hist_all.jpg", histAll);
    imwrite("orb_hist_good.jpg", histGood);

    return good.size();
}


int main() {

    //size_t goodCount = matchAndSaveSIFT_BF("in2.png", "in1.png", "sift_matches.jpg");
    //cout << "Good matches: " << goodCount << "\n";

    size_t goodCount = matchAndSaveSIFT_BF("pic11.png", "pic22.png", "sift_matches_pic11_22.jpg");
    cout << "Good matches: " << goodCount << "\n";

    //size_t orbGood = matchAndSaveORB_BF("in2.png", "in1.png", "orb_matches.jpg");
    //cout << "ORB good matches: " << orbGood << "\n";
    
    //size_t orbGood = matchAndSaveORB_BF("pic11.png", "pic22.png", "orb_matches_pic11_22.jpg");
    //cout << "ORB good matches: " << orbGood << "\n";
    
    return 0;

}
