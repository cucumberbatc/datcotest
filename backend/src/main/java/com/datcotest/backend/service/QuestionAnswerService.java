package com.datcotest.backend.service;

import com.datcotest.backend.dto.ApiDtos;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.server.ResponseStatusException;

@Service
public class QuestionAnswerService {

  static final Set<String> STOP_WORDS = Set.of(
      "the", "and", "for", "with", "that", "this", "from", "into", "then", "than",
      "what", "when", "where", "which", "while", "about", "have", "will", "were",
      "있나요", "얼마", "무엇", "관련", "대한", "에서", "으로", "이며", "그리고",
      "또한", "문서", "질문", "회사", "해당", "기준", "정리", "요약", "해주세요",
      "알려", "주세요", "있어", "있는", "하는", "적용", "범위", "핵심"
  );

  private static final Pattern MULTI_SPACE = Pattern.compile("\\s+");

  private final DocumentService documentService;

  public QuestionAnswerService(DocumentService documentService) {
    this.documentService = documentService;
  }

  public ApiDtos.AskResponse ask(ApiDtos.AskRequest request) {
    if (request == null || !StringUtils.hasText(request.question())) {
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "question is required");
    }

    Instant startedAt = Instant.now();
    List<DocumentService.StoredDocument> scopedDocuments = documentService.resolveDocuments(request.documentIds());
    if (scopedDocuments.isEmpty()) {
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "No indexed documents available");
    }

    List<String> queryTokens = documentService.tokenize(request.question());
    List<DocumentService.ChunkData> chunks = documentService.buildChunks(scopedDocuments, queryTokens);

    if (chunks.isEmpty()) {
      return new ApiDtos.AskResponse(
          request.question(),
          "",
          true,
          buildNoEvidenceNote(scopedDocuments.size()),
          List.of(),
          Duration.between(startedAt, Instant.now()).toMillis()
      );
    }

    Map<String, Integer> documentFrequencies = buildDocumentFrequencies(chunks);
    List<Candidate> ranked = chunks.stream()
        .map(chunk -> scoreChunk(request.question(), queryTokens, chunks.size(), documentFrequencies, chunk))
        .filter(candidate -> candidate.score() > 0)
        .sorted(Comparator.comparingDouble(Candidate::score).reversed())
        .toList();

    List<Candidate> selected = selectTopCandidates(ranked, 3);
    if (selected.isEmpty() || selected.get(0).score() < 1.0 || selected.get(0).overlapCount() == 0) {
      return new ApiDtos.AskResponse(
          request.question(),
          "",
          true,
          buildNoEvidenceNote(scopedDocuments.size()),
          List.of(),
          Duration.between(startedAt, Instant.now()).toMillis()
      );
    }

    List<ApiDtos.SourceResponse> sources = new ArrayList<>();
    for (int index = 0; index < selected.size(); index++) {
      Candidate candidate = selected.get(index);
      sources.add(new ApiDtos.SourceResponse(
          index + 1,
          candidate.chunk().documentId(),
          candidate.chunk().fileName(),
          candidate.chunk().pageNumber(),
          candidate.chunk().paragraphIndex(),
          "p." + candidate.chunk().pageNumber() + " ¶" + candidate.chunk().paragraphIndex(),
          abbreviate(candidate.chunk().text(), 280),
          round(candidate.score())
      ));
    }

    return new ApiDtos.AskResponse(
        request.question(),
        composeAnswer(sources),
        false,
        null,
        sources,
        Duration.between(startedAt, Instant.now()).toMillis()
    );
  }

  private Map<String, Integer> buildDocumentFrequencies(List<DocumentService.ChunkData> chunks) {
    Map<String, Integer> frequencies = new HashMap<>();
    for (DocumentService.ChunkData chunk : chunks) {
      Set<String> uniqueTokens = new HashSet<>(chunk.tokens());
      for (String token : uniqueTokens) {
        frequencies.merge(token, 1, Integer::sum);
      }
    }
    return frequencies;
  }

  private Candidate scoreChunk(
      String question,
      List<String> queryTokens,
      int corpusSize,
      Map<String, Integer> documentFrequencies,
      DocumentService.ChunkData chunk
  ) {
    Map<String, Integer> termFrequencies = new HashMap<>();
    for (String token : chunk.tokens()) {
      termFrequencies.merge(token, 1, Integer::sum);
    }

    double score = 0;
    int overlapCount = 0;
    Set<String> countedOverlap = new HashSet<>();
    for (String queryToken : queryTokens) {
      int frequency = termFrequencies.getOrDefault(queryToken, 0);
      if (frequency == 0) {
        continue;
      }

      if (countedOverlap.add(queryToken)) {
        overlapCount++;
      }

      int documentFrequency = documentFrequencies.getOrDefault(queryToken, 0);
      double idf = Math.log((corpusSize + 1.0) / (documentFrequency + 1.0)) + 1.0;
      score += (1.0 + Math.log(frequency)) * idf;
    }

    String lowerQuestion = question.toLowerCase(Locale.ROOT);
    String lowerChunk = chunk.text().toLowerCase(Locale.ROOT);

    if (containsDigit(question) && containsDigit(chunk.text())) {
      score += 0.35;
    }
    if (lowerQuestion.contains("언제") && lowerChunk.matches(".*\\d{4}.*")) {
      score += 0.25;
    }
    if ((lowerQuestion.contains("얼마") || lowerQuestion.contains("%")) && lowerChunk.matches(".*[0-9%억원만천krw₩].*")) {
      score += 0.3;
    }
    if (lowerQuestion.contains("우선순위") && lowerChunk.contains("우선순위")) {
      score += 0.4;
    }
    if (lowerQuestion.contains("적용 범위") && lowerChunk.contains("적용 범위")) {
      score += 0.4;
    }
    if (overlapCount >= 2) {
      score += 0.2;
    }

    return new Candidate(chunk, score, overlapCount);
  }

  private List<Candidate> selectTopCandidates(List<Candidate> ranked, int limit) {
    List<Candidate> selected = new ArrayList<>();
    Set<String> seenLocations = new HashSet<>();
    for (Candidate candidate : ranked) {
      String key = candidate.chunk().documentId() + ":" + candidate.chunk().pageNumber() + ":" + candidate.chunk().paragraphIndex();
      if (!seenLocations.add(key)) {
        continue;
      }
      selected.add(candidate);
      if (selected.size() == limit) {
        break;
      }
    }
    return selected;
  }

  private String composeAnswer(List<ApiDtos.SourceResponse> sources) {
    if (sources.isEmpty()) {
      return "";
    }

    StringBuilder builder = new StringBuilder("문서 기준으로 확인된 내용은 다음과 같습니다. ");
    for (int index = 0; index < sources.size(); index++) {
      ApiDtos.SourceResponse source = sources.get(index);
      if (index == 0) {
        builder.append(source.excerpt()).append(" [").append(source.citationNumber()).append("]");
      } else if (index == 1) {
        builder.append(" 추가 근거로 ").append(source.excerpt()).append(" [").append(source.citationNumber()).append("]");
      } else {
        builder.append(" 또한 ").append(source.excerpt()).append(" [").append(source.citationNumber()).append("]");
      }
    }
    return MULTI_SPACE.matcher(builder.toString()).replaceAll(" ").trim();
  }

  private String buildNoEvidenceNote(int scopedDocumentCount) {
    return scopedDocumentCount
        + "개 문서에서 질문과 직접적으로 연결되는 근거 문단을 찾지 못했습니다. 질문을 더 구체화하거나 관련 PDF를 추가해 주세요.";
  }

  private boolean containsDigit(String text) {
    return text != null && text.chars().anyMatch(Character::isDigit);
  }

  private String abbreviate(String text, int maxLength) {
    String normalized = MULTI_SPACE.matcher(text).replaceAll(" ").trim();
    if (normalized.length() <= maxLength) {
      return normalized;
    }
    return normalized.substring(0, Math.max(0, maxLength - 1)).trim() + "…";
  }

  private double round(double value) {
    return Math.round(value * 100.0) / 100.0;
  }

  private record Candidate(
      DocumentService.ChunkData chunk,
      double score,
      int overlapCount
  ) {}
}
