package com.datcotest.backend.controller;

import com.datcotest.backend.dto.ApiDtos;
import com.datcotest.backend.service.QuestionAnswerService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

  private final QuestionAnswerService questionAnswerService;

  public ChatController(QuestionAnswerService questionAnswerService) {
    this.questionAnswerService = questionAnswerService;
  }

  @PostMapping("/ask")
  public ApiDtos.AskResponse ask(@Valid @RequestBody ApiDtos.AskRequest request) {
    return questionAnswerService.ask(request);
  }
}
