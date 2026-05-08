package com.datcotest.backend.dto;

import jakarta.validation.constraints.NotBlank;
import java.util.List;

public final class ApiDtos {

  private ApiDtos() {}

  public record DocumentSummaryResponse(
      String id,
      String fileName,
      long sizeBytes,
      String uploadedAt,
      String status,
      int pageCount
  ) {}

  public record PageResponse(
      int pageNumber,
      List<String> paragraphs
  ) {}

  public record DocumentDetailResponse(
      String id,
      String fileName,
      long sizeBytes,
      String uploadedAt,
      String status,
      int pageCount,
      String downloadUrl,
      List<PageResponse> pages
  ) {}

  public record UploadResponse(
      List<DocumentSummaryResponse> documents
  ) {}

  public record AskRequest(
      @NotBlank(message = "question is required")
      String question,
      List<String> documentIds
  ) {}

  public record SourceResponse(
      int citationNumber,
      String documentId,
      String fileName,
      int pageNumber,
      int paragraphIndex,
      String locationLabel,
      String excerpt,
      double score
  ) {}

  public record AskResponse(
      String question,
      String answer,
      boolean noEvidence,
      String noEvidenceNote,
      List<SourceResponse> sources,
      long elapsedMs
  ) {}
}
