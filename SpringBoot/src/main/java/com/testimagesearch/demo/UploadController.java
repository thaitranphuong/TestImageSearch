package com.testimagesearch.demo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

@RestController
@CrossOrigin
public class UploadController {
	
	@Autowired
	ImageRepository imageRepository;

	@PostMapping("/search")
	public Vectors[] saveFile(@RequestPart("main_image") MultipartFile mainImage) throws IOException {
		if(mainImage.isEmpty()) return null;
		
		Path path = Paths.get("uploads/");
		InputStream inputStream = mainImage.getInputStream();
		
		ImageEntity imageEntity = new ImageEntity();
		imageEntity.setName(mainImage.getOriginalFilename());
		
		// Thêm mainImage vào thư mục
		Files.copy(inputStream, path.resolve(mainImage.getOriginalFilename()), StandardCopyOption.REPLACE_EXISTING);
		
		// Gửi HTTP request tới Flask server để kích hoạt search
        String flaskServerUrl = "http://localhost:5000/find-similar-images"; // Đổi địa chỉ tùy vào Flask server của bạn
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<Vectors[]> responseEntity = restTemplate.postForEntity(flaskServerUrl, 
        		Paths.get("uploads/").toAbsolutePath().normalize().toString() + "\\" + mainImage.getOriginalFilename(),
        		Vectors[].class);
        
     // Lấy giá trị trả về từ response
        Vectors[] vectors = responseEntity.getBody();
        return vectors;
	}
	
	@PostMapping("/retrain")
	public void retrain(@RequestPart("main_image") MultipartFile mainImage) throws IOException {
		if(mainImage.isEmpty()) return;
		
		Path path = Paths.get("dataset/");
		InputStream inputStream = mainImage.getInputStream();
		
		ImageEntity imageEntity = new ImageEntity();
		imageEntity.setName(mainImage.getOriginalFilename());
		
		String id = imageRepository.save(imageEntity).getId().toString();
		
		// Thêm mainImage vào thư mục
		Files.copy(inputStream, path.resolve(id + ".png"), StandardCopyOption.REPLACE_EXISTING);
		
		// Gửi HTTP request tới Flask server để kích hoạt đào tạo lại mô hình
        String flaskServerUrl = "http://localhost:5000/retrain"; // Đổi địa chỉ tùy vào Flask server của bạn
        RestTemplate restTemplate = new RestTemplate();
        restTemplate.postForObject(flaskServerUrl, 
        		Paths.get("dataset").toAbsolutePath().normalize().toString() + "\\" + id + ".png",
        		String.class);
	}
	
	@GetMapping("/getimage/{image}")
	public ResponseEntity<ByteArrayResource> getImage(@PathVariable("image") String image) {
		if(!image.equals("") || image != null) {
			try {
				Path filename = Paths.get("dataset", image);
				byte[] buffer = Files.readAllBytes(filename);
				ByteArrayResource byteArrayResource = new ByteArrayResource(buffer);
				return ResponseEntity.ok()
						.contentLength(buffer.length)
						.contentType(MediaType.parseMediaType("image/png"))
						.body(byteArrayResource);
			} catch (Exception e) {
				
			}
		}
		return ResponseEntity.badRequest().build();
	}
	
	
}
