package com.datcotest.backend.controller;

import com.datcotest.backend.dto.ApiDtos;
import com.datcotest.backend.service.DocumentService;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@Validated
@RestController
@RequestMapping("/api/documents")
public class DocumentController {

  private final DocumentService documentService;

  public DocumentController(DocumentService documentService) {
    this.documentService = documentService;
  }

  @GetMapping
  public List<ApiDtos.DocumentSummaryResponse> listDocuments() {
    return documentService.listDocumentSummaries();
  }

  @GetMapping("/{documentId}")
  public ApiDtos.DocumentDetailResponse getDocument(@PathVariable String documentId) {
    return documentService.getDocumentDetail(documentId);
  }

  @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
  public ApiDtos.UploadResponse uploadDocuments(@RequestParam("files") List<MultipartFile> files) {
    return documentService.upload(files);
  }

  @DeleteMapping("/{documentId}")
  public void deleteDocument(@PathVariable String documentId) {
    documentService.delete(documentId);
  }

  @GetMapping("/{documentId}/file")
  public ResponseEntity<Resource> downloadDocument(@PathVariable String documentId) throws IOException {
    Path filePath = documentService.getStoredFile(documentId);
    Resource resource = new FileSystemResource(filePath);

    return ResponseEntity.ok()
        .contentType(MediaType.APPLICATION_PDF)
        .contentLength(Files.size(filePath))
        .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + filePath.getFileName() + "\"")
        .body(resource);
  }
}
