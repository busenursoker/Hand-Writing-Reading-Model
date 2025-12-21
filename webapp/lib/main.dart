import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:web/web.dart' as web;

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EL Yazısı Tanıma Modeli',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        scaffoldBackgroundColor: const Color(0xFFF7F6FB),
      ),
      home: const PredictWebScreen(),
    );
  }
}

class PredictWebScreen extends StatefulWidget {
  const PredictWebScreen({super.key});

  @override
  State<PredictWebScreen> createState() => _PredictWebScreenState();
}

class _PredictWebScreenState extends State<PredictWebScreen> {
  Uint8List? _imageBytes;
  String? _prediction;
  String? _error;
  bool _loading = false;

  final String baseUrl = "http://localhost:8000";

  Future<Uint8List?> _pickImageBytesWeb() async {
    final input = web.HTMLInputElement()
      ..type = 'file'
      ..accept = 'image/png,image/jpeg';

    web.document.body?.append(input);
    input.click();

    try {
      await input.onChange.first.timeout(const Duration(seconds: 60));

      final file = input.files?.item(0);
      if (file == null) return null;

      final mime = (file.type).toLowerCase();
      if (mime != 'image/png' && mime != 'image/jpeg') {
        throw Exception("Lütfen PNG veya JPG/JPEG seç (HEIC/WebP desteklenmiyor).");
      }

      final reader = web.FileReader();
      final completer = Completer<Uint8List?>();

      reader.onLoadEnd.listen((_) {
        final result = reader.result;
        if (result == null) {
          if (!completer.isCompleted) completer.complete(null);
          return;
        }

        final dataUrl = result.toString();
        final comma = dataUrl.indexOf(',');
        if (comma < 0) {
          if (!completer.isCompleted) completer.complete(null);
          return;
        }

        final b64 = dataUrl.substring(comma + 1);
        try {
          final bytes = base64Decode(b64);
          if (!completer.isCompleted) completer.complete(bytes);
        } catch (_) {
          if (!completer.isCompleted) {
            completer.completeError(Exception("Görsel base64 decode edilemedi."));
          }
        }
      });

      reader.readAsDataURL(file);

      return completer.future.timeout(
        const Duration(seconds: 10),
        onTimeout: () => null,
      );
    } on TimeoutException {
      return null;
    } finally {
      input.remove();
    }
  }

  Future<void> pickImage() async {
    setState(() {
      _error = null;
      _prediction = null;
    });

    try {
      final bytes = await _pickImageBytesWeb();
      if (bytes == null) return;

      setState(() => _imageBytes = bytes);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _imageBytes = null;
      });
    }
  }

  Future<void> predict() async {
    if (_imageBytes == null) return;

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
    });

    try {
      final uri = Uri.parse("$baseUrl/predict-word?infer_orientation=auto");
      final req = http.MultipartRequest("POST", uri);

      req.files.add(
        http.MultipartFile.fromBytes(
          "image",
          _imageBytes!,
          filename: "upload.jpg",
        ),
      );

      final streamed = await req.send();
      final resp = await http.Response.fromStream(streamed);

      if (resp.statusCode != 200) {
        throw Exception("Server error: ${resp.statusCode}\n${resp.body}");
      }

      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      setState(() => _prediction = (data["prediction"] ?? "").toString());
    } catch (e) {
      setState(() {
        _error =
            "İstek atılamadı (Failed to fetch). Muhtemelen CORS/Backend URL.\n"
            "URL: $baseUrl\n"
            "Hata: $e";
      });
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final canPredict = !_loading && _imageBytes != null;

    final preview = Container(
      height: 280,
      width: double.infinity,
      decoration: BoxDecoration(
        color: cs.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: _imageBytes == null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.cloud_upload_outlined, size: 44, color: cs.primary),
                  const SizedBox(height: 10),
                  Text("PNG / JPG görsel seç", style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 6),
                  Text(
                    "El yazısı kelime fotoğrafını yükle",
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.outline),
                  ),
                ],
              ),
            )
          : ClipRRect(
              borderRadius: BorderRadius.circular(18),
              child: Stack(
                children: [
                  Positioned.fill(child: Image.memory(_imageBytes!, fit: BoxFit.contain)),
                  Positioned(
                    right: 12,
                    top: 12,
                    child: FilledButton.tonalIcon(
                      onPressed: _loading ? null : pickImage,
                      icon: const Icon(Icons.refresh),
                      label: const Text("Değiştir"),
                    ),
                  ),
                ],
              ),
            ),
    );

    Widget resultWidget() {
      if (_error != null) {
        return Container(
          margin: const EdgeInsets.only(top: 14),
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: cs.errorContainer,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: cs.error.withOpacity(0.25)),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.error_outline, color: cs.error),
              const SizedBox(width: 10),
              Expanded(child: Text(_error!, style: TextStyle(color: cs.onErrorContainer))),
            ],
          ),
        );
      }

      if (_prediction != null) {
        return Card(
          margin: const EdgeInsets.only(top: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                CircleAvatar(
                  backgroundColor: cs.primaryContainer,
                  child: Icon(Icons.text_fields, color: cs.onPrimaryContainer),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    _prediction!,
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                ),
              ],
            ),
          ),
        );
      }

      return const SizedBox.shrink();
    }

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              cs.primary.withOpacity(0.08),
              cs.secondary.withOpacity(0.06),
              cs.surface,
            ],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 920),
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Card(
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "EL Yazısı Tanıma Modeli",
                          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                fontWeight: FontWeight.w700,
                              ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          "Görsel yükle → modeli çalıştır → sonucu gör",
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: cs.outline),
                        ),
                        const SizedBox(height: 16),
                        if (_loading) const LinearProgressIndicator(),
                        if (_loading) const SizedBox(height: 12),
                        preview,
                        const SizedBox(height: 14),
                        Row(
                          children: [
                            Expanded(
                              child: FilledButton.icon(
                                onPressed: _loading ? null : pickImage,
                                icon: const Icon(Icons.upload_file),
                                label: const Text("Görsel Seç"),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: FilledButton.tonalIcon(
                                onPressed: canPredict ? predict : null,
                                icon: Icon(_loading ? Icons.hourglass_top : Icons.play_arrow),
                                label: Text(_loading ? "Tahmin ediliyor..." : "Tahmin Et"),
                              ),
                            ),
                          ],
                        ),
                        resultWidget(),
                        const SizedBox(height: 14),
                        Row(
                          children: [
                            Icon(Icons.link, size: 18, color: cs.outline),
                            const SizedBox(width: 6),
                            Text(
                              "Backend: $baseUrl",
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.outline),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
