package com.datcotest.backend.service;

import com.datcotest.backend.dto.ApiDtos;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

@Service
public class DocumentService {

  private static final Pattern WHITESPACE = Pattern.compile("\\s+");
  private static final Pattern SAFE_FILE_CHARS = Pattern.compile("[^a-zA-Z0-9._-]");
  private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ISO_INSTANT;

  private final Path storageRoot;
  private final Map<String, StoredDocument> documents = new ConcurrentHashMap<>();

  public DocumentService(@Value("${app.storage-root:./backend-storage}") String storageRoot) {
    try {
      this.storageRoot = Path.of(storageRoot).toAbsolutePath().normalize();
      Files.createDirectories(this.storageRoot);
    } catch (IOException exception) {
      throw new UncheckedIOException("Failed to initialize storage directory", exception);
    }
  }

  public List<ApiDtos.DocumentSummaryResponse> listDocumentSummaries() {
    return documents.values().stream()
        .sorted(Comparator.comparing(StoredDocument::uploadedAt).reversed())
        .map(this::toSummary)
        .toList();
  }

  public ApiDtos.DocumentDetailResponse getDocumentDetail(String documentId) {
    return toDetail(getDocument(documentId));
  }

  public StoredDocument getDocument(String documentId) {
    StoredDocument document = documents.get(documentId);
    if (document == null) {
      throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Document not found: " + documentId);
    }
    return document;
  }

  public List<StoredDocument> resolveDocuments(List<String> documentIds) {
    if (documentIds == null || documentIds.isEmpty()) {
      return documents.values().stream()
          .sorted(Comparator.comparing(StoredDocument::uploadedAt).reversed())
          .toList();
    }

    List<StoredDocument> resolved = new ArrayList<>();
    for (String documentId : new LinkedHashSet<>(documentIds)) {
      resolved.add(getDocument(documentId));
    }
    return resolved;
  }

  public ApiDtos.UploadResponse upload(List<MultipartFile> files) {
    if (files == null || files.isEmpty()) {
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "At least one PDF file is required");
    }

    List<ApiDtos.DocumentSummaryResponse> uploaded = new ArrayList<>();
    for (MultipartFile file : files) {
      if (file.isEmpty()) {
        continue;
      }
      uploaded.add(toSummary(ingest(file)));
    }

    if (uploaded.isEmpty()) {
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "All uploaded files were empty");
    }
    return new ApiDtos.UploadResponse(uploaded);
  }

  public void delete(String documentId) {
    StoredDocument removed = documents.remove(documentId);
    if (removed == null) {
      throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Document not found: " + documentId);
    }

    try {
      Files.deleteIfExists(removed.filePath());
    } catch (IOException exception) {
      throw new UncheckedIOException("Failed to delete stored PDF file", exception);
    }
  }

  public Path getStoredFile(String documentId) {
    return getDocument(documentId).filePath();
  }

  public List<ChunkData> buildChunks(List<StoredDocument> scopedDocuments, List<String> queryTokens) {
    Set<String> queryTokenSet = Set.copyOf(queryTokens);
    List<ChunkData> chunks = new ArrayList<>();

    for (StoredDocument document : scopedDocuments) {
      for (PageData page : document.pages()) {
        for (int paragraphIndex = 0; paragraphIndex < page.paragraphs().size(); paragraphIndex++) {
          String paragraph = page.paragraphs().get(paragraphIndex);
          List<String> tokens = tokenize(paragraph);
          if (tokens.isEmpty()) {
            continue;
          }

          if (!queryTokenSet.isEmpty() && queryTokenSet.stream().noneMatch(tokens::contains)) {
            continue;
          }

          chunks.add(new ChunkData(
              document.id(),
              document.fileName(),
              page.pageNumber(),
              paragraphIndex + 1,
              paragraph,
              tokens
          ));
        }
      }
    }

    if (!chunks.isEmpty() || queryTokens.isEmpty()) {
      return chunks;
    }

    for (StoredDocument document : scopedDocuments) {
      for (PageData page : document.pages()) {
        for (int paragraphIndex = 0; paragraphIndex < page.paragraphs().size(); paragraphIndex++) {
          String paragraph = page.paragraphs().get(paragraphIndex);
          List<String> tokens = tokenize(paragraph);
          if (tokens.isEmpty()) {
            continue;
          }
          chunks.add(new ChunkData(
              document.id(),
              document.fileName(),
              page.pageNumber(),
              paragraphIndex + 1,
              paragraph,
              tokens
          ));
        }
      }
    }
    return chunks;
  }

  public List<String> tokenize(String text) {
    if (!StringUtils.hasText(text)) {
      return List.of();
    }

    return Pattern.compile("[\\p{L}\\p{N}]+")
        .matcher(text.toLowerCase(Locale.ROOT))
        .results()
        .map(result -> result.group().trim())
        .filter(token -> token.length() > 1)
        .filter(token -> !QuestionAnswerService.STOP_WORDS.contains(token))
        .toList();
  }

  private StoredDocument ingest(MultipartFile file) {
    String originalName = Optional.ofNullable(file.getOriginalFilename())
        .map(StringUtils::cleanPath)
        .filter(StringUtils::hasText)
        .orElse("document.pdf");

    if (!originalName.toLowerCase(Locale.ROOT).endsWith(".pdf")) {
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Only PDF uploads are supported: " + originalName);
    }

    String id = UUID.randomUUID().toString();
    String safeName = SAFE_FILE_CHARS
        .matcher(originalName.replace(" ", "_"))
        .replaceAll("");
    if (!StringUtils.hasText(safeName)) {
      safeName = "document.pdf";
    }

    Path destination = storageRoot.resolve(id + "-" + safeName).normalize();
    try {
      Files.copy(file.getInputStream(), destination, StandardCopyOption.REPLACE_EXISTING);
      List<PageData> pages = extractPages(destination);
      StoredDocument document = new StoredDocument(
          id,
          originalName,
          Files.size(destination),
          Instant.now(),
          "ready",
          pages.size(),
          pages,
          destination
      );
      documents.put(document.id(), document);
      return document;
    } catch (IOException exception) {
      throw new UncheckedIOException("Failed to ingest PDF: " + originalName, exception);
    }
  }

  private List<PageData> extractPages(Path path) throws IOException {
    try (PDDocument document = Loader.loadPDF(path.toFile())) {
      PDFTextStripper stripper = new PDFTextStripper();
      List<PageData> pages = new ArrayList<>();

      for (int pageNumber = 1; pageNumber <= document.getNumberOfPages(); pageNumber++) {
        stripper.setStartPage(pageNumber);
        stripper.setEndPage(pageNumber);
        String pageText = stripper.getText(document);
        List<String> paragraphs = splitParagraphs(pageText);
        pages.add(new PageData(pageNumber, paragraphs));
      }

      return pages;
    }
  }

  private List<String> splitParagraphs(String rawPageText) {
    String normalized = rawPageText == null ? "" : rawPageText.replace("\r", "\n");
    String[] blocks = normalized.split("\\n\\s*\\n");
    List<String> paragraphs = new ArrayList<>();

    for (String block : blocks) {
      String cleaned = normalizeText(block);
      if (!StringUtils.hasText(cleaned)) {
        continue;
      }
      if (cleaned.length() <= 500) {
        paragraphs.add(cleaned);
        continue;
      }

      List<String> sentences = Arrays.stream(cleaned.split("(?<=[.!?。])\\s+"))
          .map(this::normalizeText)
          .filter(StringUtils::hasText)
          .toList();

      if (sentences.isEmpty()) {
        paragraphs.add(cleaned);
        continue;
      }

      StringBuilder buffer = new StringBuilder();
      for (String sentence : sentences) {
        if (buffer.length() > 0 && buffer.length() + sentence.length() > 420) {
          paragraphs.add(buffer.toString().trim());
          buffer.setLength(0);
        }
        if (buffer.length() > 0) {
          buffer.append(' ');
        }
        buffer.append(sentence);
      }
      if (buffer.length() > 0) {
        paragraphs.add(buffer.toString().trim());
      }
    }

    if (!paragraphs.isEmpty()) {
      return paragraphs;
    }

    return List.of("이 페이지에서는 추출 가능한 텍스트를 찾지 못했습니다.");
  }

  private String normalizeText(String text) {
    return WHITESPACE.matcher(text == null ? "" : text).replaceAll(" ").trim();
  }

  private ApiDtos.DocumentSummaryResponse toSummary(StoredDocument document) {
    return new ApiDtos.DocumentSummaryResponse(
        document.id(),
        document.fileName(),
        document.sizeBytes(),
        ISO_FORMATTER.format(document.uploadedAt()),
        document.status(),
        document.pageCount()
    );
  }

  private ApiDtos.DocumentDetailResponse toDetail(StoredDocument document) {
    return new ApiDtos.DocumentDetailResponse(
        document.id(),
        document.fileName(),
        document.sizeBytes(),
        ISO_FORMATTER.format(document.uploadedAt()),
        document.status(),
        document.pageCount(),
        "/api/documents/" + document.id() + "/file",
        document.pages().stream()
            .map(page -> new ApiDtos.PageResponse(page.pageNumber(), page.paragraphs()))
            .toList()
    );
  }

  public record StoredDocument(
      String id,
      String fileName,
      long sizeBytes,
      Instant uploadedAt,
      String status,
      int pageCount,
      List<PageData> pages,
      Path filePath
  ) {}

  public record PageData(
      int pageNumber,
      List<String> paragraphs
  ) {}

  public record ChunkData(
      String documentId,
      String fileName,
      int pageNumber,
      int paragraphIndex,
      String text,
      List<String> tokens
  ) {}
}
