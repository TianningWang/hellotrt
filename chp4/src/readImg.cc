#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

using fs = std::filesystem;

std::vector<cv::Mat> readJpgImages(const std::string folderPath)
{
	std::vector<cv::Mat> images;

	for (const auto& entry : fs::directory_iterator(folderPath))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".jpg")
		{
			std::string imagePath = entry.path().string();
			cv::Mat image = cv::imread(imagePath);

			if (!image.empty())
			{
				images.push_back(image);
			}
		}
	}
	return images;
}

int main()
{
	std::string folderPath = "folder_path";
	std::vector<cv::Mat> images = readJpgImages(folderPath);

	std::cout << "read images count: " << images.size() << std::endl;
	return 0;
}
