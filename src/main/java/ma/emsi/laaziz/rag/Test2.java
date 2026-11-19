package main.java.ma.emsi.laaziz.rag;

import java.util.List;
import java.util.Scanner;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.laaziz.rag.Assistant;

public class Test2 {

    public static void main(String[] args) {

        String apiKey = System.getProperty("GEMINI_API_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("Erreur : propriété GEMINI_API_KEY absente.");
            return;
        }


        final boolean utiliserTransformer = true;
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        DocumentParser parser = new ApacheTikaDocumentParser();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf", parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Segments enregistrés : " + segments.size());

        ContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant;

        if (utiliserTransformer) {

            // Reformule la question avec le contexte de la conversation.
            CompressingQueryTransformer queryTransformer =
                    new CompressingQueryTransformer(model);

            RetrievalAugmentor augmentor =
                    DefaultRetrievalAugmentor.builder()
                            .queryTransformer(queryTransformer)
                            .contentRetriever(retriever)
                            .build();

            assistant = AiServices.builder(Assistant.class)
                    .chatModel(model)
                    .chatMemory(memory)
                    .retrievalAugmentor(augmentor)
                    .build();

        } else {

            // Configuration initiale pour comparer les réponses.
            assistant = AiServices.builder(Assistant.class)
                    .chatModel(model)
                    .chatMemory(memory)
                    .contentRetriever(retriever)
                    .build();
        }

        System.out.println("Test 2 - QueryTransformer : "
                + (utiliserTransformer ? "ACTIVÉ" : "DÉSACTIVÉ"));

        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.print("\nQuestion : ");

                if (!scanner.hasNextLine()) {
                    break;
                }

                String question = scanner.nextLine().trim();

                if ("exit".equalsIgnoreCase(question)) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                try {
                    String reponse = assistant.chat(question);
                    System.out.println("\nRéponse : " + reponse);
                } catch (RuntimeException e) {
                    System.err.println(
                            "Erreur pendant la requête : " + e.getMessage());
                }
            }
        }
    }
}