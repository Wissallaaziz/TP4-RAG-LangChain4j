package ma.emsi.laaziz.rag;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

public class Test4 {

    public static void main(String[] args) {

        String geminiKey = System.getProperty("GEMINI_API_KEY");
        String tavilyKey = System.getProperty("TAVILY_API_KEY");

        if (geminiKey == null || geminiKey.isBlank()) {
            System.err.println("Erreur : clé Gemini absente.");
            return;
        }

        if (tavilyKey == null || tavilyKey.isBlank()) {
            System.err.println("Erreur : clé Tavily absente.");
            return;
        }


        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        DocumentParser parser = new ApacheTikaDocumentParser();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf", parser);

        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println(
                "Segments enregistrés : " + segments.size());


        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();


        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(tavilyKey)
                        .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();


        Map<ContentRetriever, String> descriptions =
                new LinkedHashMap<>();

        descriptions.put(
                pdfRetriever,
                "Support de cours privé sur l'intelligence artificielle, " +
                        "les modèles de langage, le fine-tuning, les embeddings " +
                        "et le RAG. Utiliser pour les questions sur ce cours."
        );

        descriptions.put(
                webRetriever,
                "Recherche sur Internet avec Tavily. Utiliser pour " +
                        "les actualités récentes, les événements récents, " +
                        "les informations à jour et les questions sans rapport " +
                        "avec le support de cours sur l'intelligence artificielle."
        );

        QueryRouter router =
                new LanguageModelQueryRouter(model, descriptions);

        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(router)
                        .build();

        ChatMemory memory =
                MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(memory)
                        .retrievalAugmentor(augmentor)
                        .build();

        System.out.println("Test 5 - Routage PDF / Tavily prêt.");
        System.out.println("Tapez 'exit' pour quitter.");

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
                            "Erreur pendant la requête : "
                                    + e.getMessage());
                }
            }
        }
    }
}