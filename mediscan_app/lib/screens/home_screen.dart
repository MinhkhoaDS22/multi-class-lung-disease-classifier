import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:file_picker/file_picker.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

// ==========================================
// CONSTANTS
// ==========================================
const String kApiBase = 'http://localhost:8000';

const Color kBgDark = Color(0xFF050E1A);
const Color kBgCard = Color(0xFF0A1628);
const Color kAccent = Color(0xFF00B4D8);
const Color kGreen = Color(0xFF2ED573);
const Color kRed = Color(0xFFFF4757);
const Color kOrange = Color(0xFFFFA502);

// Helper: replaces deprecated withOpacity
extension ColorX on Color {
  Color o(double opacity) => withValues(alpha: opacity);
}

// ==========================================
// MODEL
// ==========================================
class DiseaseResult {
  final String classKey;
  final String nameVi;
  final String description;
  final String severity;
  final String color;
  final double probability;
  final double percentage;

  DiseaseResult({
    required this.classKey,
    required this.nameVi,
    required this.description,
    required this.severity,
    required this.color,
    required this.probability,
    required this.percentage,
  });

  factory DiseaseResult.fromJson(Map<String, dynamic> json) {
    return DiseaseResult(
      classKey: json['class_key'] ?? '',
      nameVi: json['name_vi'] ?? '',
      description: json['description'] ?? '',
      severity: json['severity'] ?? 'normal',
      color: json['color'] ?? '#00B4D8',
      probability: (json['probability'] ?? 0.0).toDouble(),
      percentage: (json['percentage'] ?? 0.0).toDouble(),
    );
  }

  Color get colorValue {
    final hex = color.replaceAll('#', '');
    return Color(int.parse('FF$hex', radix: 16));
  }
}

class PredictionResponse {
  final bool success;
  final String predictedClass;
  final String predictedNameVi;
  final String predictedSeverity;
  final double confidence;
  final List<DiseaseResult> results;

  PredictionResponse({
    required this.success,
    required this.predictedClass,
    required this.predictedNameVi,
    required this.predictedSeverity,
    required this.confidence,
    required this.results,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    return PredictionResponse(
      success: json['success'] ?? false,
      predictedClass: json['predicted_class'] ?? '',
      predictedNameVi: json['predicted_name_vi'] ?? '',
      predictedSeverity: json['predicted_severity'] ?? 'normal',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      results: (json['results'] as List<dynamic>? ?? [])
          .map((e) => DiseaseResult.fromJson(e))
          .toList(),
    );
  }
}

// ==========================================
// HOME SCREEN
// ==========================================
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  Uint8List? _imageBytes;
  String? _fileName;
  bool _isAnalyzing = false;
  PredictionResponse? _result;
  String? _errorMessage;

  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    _pulseAnimation =
        Tween<double>(begin: 0.95, end: 1.05).animate(_pulseController);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
    );
    if (result != null && result.files.first.bytes != null) {
      setState(() {
        _imageBytes = result.files.first.bytes;
        _fileName = result.files.first.name;
        _result = null;
        _errorMessage = null;
      });
    }
  }

  Future<void> _analyze() async {
    if (_imageBytes == null) return;

    setState(() {
      _isAnalyzing = true;
      _result = null;
      _errorMessage = null;
    });

    try {
      final uri = Uri.parse('$kApiBase/predict');
      final request = http.MultipartRequest('POST', uri);

      final ext = (_fileName ?? 'image.jpg').split('.').last.toLowerCase();
      final mimeType = ext == 'png' ? 'image/png' : 'image/jpeg';

      request.files.add(http.MultipartFile.fromBytes(
        'file',
        _imageBytes!,
        filename: _fileName ?? 'xray.jpg',
        contentType: MediaType.parse(mimeType),
      ));

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final json = jsonDecode(utf8.decode(response.bodyBytes));
        setState(() {
          _result = PredictionResponse.fromJson(json);
        });
      } else {
        final json = jsonDecode(utf8.decode(response.bodyBytes));
        setState(() {
          _errorMessage = json['detail'] ?? 'Lỗi không xác định từ server.';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage =
            'Không thể kết nối đến server AI.\nVui lòng kiểm tra backend đang chạy tại $kApiBase';
      });
    } finally {
      setState(() => _isAnalyzing = false);
    }
  }

  void _reset() {
    setState(() {
      _imageBytes = null;
      _fileName = null;
      _result = null;
      _errorMessage = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBgDark,
      body: Stack(
        children: [
          _buildBackgroundBlobs(),
          SingleChildScrollView(
            child: Column(
              children: [
                _buildNavBar(),
                _buildHeroBanner(),
                _buildMainContent(),
                _buildFooter(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Background blobs ─────────────────────────────────────────
  Widget _buildBackgroundBlobs() {
    return Positioned.fill(
      child: IgnorePointer(
        child: Stack(
          children: [
            Positioned(
              top: -120,
              left: -100,
              child: _blob(400, const Color(0xFF00B4D8).o(0.08)),
            ),
            Positioned(
              top: 300,
              right: -150,
              child: _blob(500, const Color(0xFF0066FF).o(0.06)),
            ),
            Positioned(
              bottom: 100,
              left: -80,
              child: _blob(350, const Color(0xFF00FFD1).o(0.05)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _blob(double size, Color color) => Container(
        width: size,
        height: size,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color),
      );

  // ── NavBar ───────────────────────────────────────────────────
  Widget _buildNavBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
      decoration: BoxDecoration(
        color: kBgCard.o(0.8),
        border: Border(bottom: BorderSide(color: kAccent.o(0.15), width: 1)),
      ),
      child: Row(
        children: [
          AnimatedBuilder(
            animation: _pulseAnimation,
            builder: (ctx, child) =>
                Transform.scale(scale: _pulseAnimation.value, child: child),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: const LinearGradient(
                  colors: [Color(0xFF00B4D8), Color(0xFF0066FF)],
                ),
                boxShadow: [
                  BoxShadow(
                      color: kAccent.o(0.4), blurRadius: 12, spreadRadius: 2)
                ],
              ),
              child: const Icon(Icons.medical_services_rounded,
                  color: Colors.white, size: 22),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            'MediScan AI',
            style: GoogleFonts.inter(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: Colors.white,
              letterSpacing: -0.5,
            ),
          ),
          const Spacer(),
          _navChip(Icons.shield_rounded, 'Bảo mật'),
          const SizedBox(width: 8),
          _navChip(Icons.speed_rounded, '91.76% Accuracy'),
        ],
      ),
    );
  }

  Widget _navChip(IconData icon, String label) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: kAccent.o(0.08),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: kAccent.o(0.2)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: kAccent),
            const SizedBox(width: 6),
            Text(label,
                style: GoogleFonts.inter(
                    fontSize: 12, color: kAccent, fontWeight: FontWeight.w500)),
          ],
        ),
      );

  // ── Hero Banner ──────────────────────────────────────────────
  Widget _buildHeroBanner() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 60, horizontal: 24),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: kAccent.o(0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: kAccent.o(0.3)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                    width: 8,
                    height: 8,
                    decoration: const BoxDecoration(
                        shape: BoxShape.circle, color: kGreen)),
                const SizedBox(width: 8),
                Text('Powered by EfficientNet-B5 · TTA×3',
                    style: GoogleFonts.inter(
                        fontSize: 12,
                        color: kAccent,
                        fontWeight: FontWeight.w500)),
              ],
            ),
          ).animate().fadeIn(duration: 600.ms).slideY(begin: -0.3, end: 0),
          const SizedBox(height: 24),
          Text(
            'Chẩn đoán X-quang\nngực bằng AI',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 52,
              fontWeight: FontWeight.w800,
              color: Colors.white,
              height: 1.15,
              letterSpacing: -1.5,
            ),
          )
              .animate()
              .fadeIn(delay: 200.ms, duration: 700.ms)
              .slideY(begin: 0.2, end: 0),
          const SizedBox(height: 20),
          ShaderMask(
            shaderCallback: (bounds) => const LinearGradient(
              colors: [
                Color(0xFF00B4D8),
                Color(0xFF0066FF),
                Color(0xFF00FFD1)
              ],
            ).createShader(bounds),
            child: Text(
              'COVID · Xơ phổi · Mờ phổi · Bình thường · Viêm phổi virus',
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: Colors.white),
            ),
          )
              .animate()
              .fadeIn(delay: 400.ms, duration: 700.ms)
              .slideY(begin: 0.2, end: 0),
          const SizedBox(height: 32),
          _buildStatsRow()
              .animate()
              .fadeIn(delay: 600.ms, duration: 700.ms)
              .slideY(begin: 0.3, end: 0),
        ],
      ),
    );
  }

  Widget _buildStatsRow() => Wrap(
        spacing: 16,
        runSpacing: 16,
        alignment: WrapAlignment.center,
        children: [
          _statCard('91.76%', 'Độ chính xác', Icons.verified_rounded, kGreen),
          _statCard('5', 'Loại bệnh', Icons.category_rounded, kAccent),
          _statCard('TTA×3', 'Tăng cường dự đoán',
              Icons.auto_awesome_rounded, kOrange),
          _statCard('< 3s', 'Thời gian phân tích', Icons.timer_rounded,
              const Color(0xFFBF5AF2)),
        ],
      );

  Widget _statCard(
      String value, String label, IconData icon, Color accentColor) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        color: kBgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: accentColor.o(0.2)),
        boxShadow: [
          BoxShadow(
              color: accentColor.o(0.08), blurRadius: 20, spreadRadius: 2)
        ],
      ),
      child: Column(
        children: [
          Icon(icon, color: accentColor, size: 22),
          const SizedBox(height: 8),
          Text(value,
              style: GoogleFonts.inter(
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                  letterSpacing: -0.5)),
          const SizedBox(height: 2),
          Text(label,
              style: GoogleFonts.inter(
                  fontSize: 11,
                  color: Colors.white54,
                  fontWeight: FontWeight.w400)),
        ],
      ),
    );
  }

  // ── Main Content ─────────────────────────────────────────────
  Widget _buildMainContent() {
    final isWide = MediaQuery.of(context).size.width > 900;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
      constraints: const BoxConstraints(maxWidth: 1200),
      child: isWide
          ? Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: _buildUploadPanel()),
                const SizedBox(width: 24),
                Expanded(child: _buildResultPanel()),
              ],
            )
          : Column(children: [
              _buildUploadPanel(),
              const SizedBox(height: 24),
              _buildResultPanel(),
            ]),
    );
  }

  // ── Upload Panel ─────────────────────────────────────────────
  Widget _buildUploadPanel() {
    return _glassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(Icons.upload_file_rounded, 'Tải lên ảnh X-quang'),
          const SizedBox(height: 20),
          _buildDropZone(),
          if (_imageBytes != null) ...[
            const SizedBox(height: 16),
            _buildImagePreview(),
            const SizedBox(height: 20),
            _buildAnalyzeButton(),
          ],
          if (_imageBytes == null) ...[
            const SizedBox(height: 20),
            _buildFormatHint(),
          ],
        ],
      ),
    ).animate().fadeIn(delay: 200.ms, duration: 600.ms).slideX(begin: -0.1, end: 0);
  }

  Widget _buildDropZone() {
    return GestureDetector(
      onTap: _pickImage,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: Container(
          height: 200,
          decoration: BoxDecoration(
            color: kAccent.o(0.04),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: kAccent.o(0.35), width: 2),
          ),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  padding: const EdgeInsets.all(18),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [kAccent.o(0.2), kAccent.o(0.0)],
                    ),
                  ),
                  child: Icon(Icons.add_photo_alternate_rounded,
                      size: 48, color: kAccent),
                ),
                const SizedBox(height: 14),
                Text('Nhấn để chọn ảnh',
                    style: GoogleFonts.inter(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.white)),
                const SizedBox(height: 4),
                Text('JPG, PNG, WEBP – tối đa 20MB',
                    style:
                        GoogleFonts.inter(fontSize: 12, color: Colors.white38)),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.image_rounded, color: kAccent, size: 18),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                _fileName ?? 'Ảnh đã chọn',
                style: GoogleFonts.inter(
                    fontSize: 13,
                    color: Colors.white70,
                    fontWeight: FontWeight.w500),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            GestureDetector(
              onTap: _reset,
              child:
                  Icon(Icons.close_rounded, color: Colors.white38, size: 18),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Container(
            constraints: const BoxConstraints(maxHeight: 320),
            width: double.infinity,
            child: Image.memory(_imageBytes!, fit: BoxFit.contain),
          ),
        ),
      ],
    );
  }

  Widget _buildAnalyzeButton() {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: _isAnalyzing ? _loadingButton() : _analyzeButton(),
    );
  }

  Widget _analyzeButton() {
    return ElevatedButton(
      onPressed: _analyze,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.transparent,
        shadowColor: Colors.transparent,
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        padding: EdgeInsets.zero,
      ),
      child: Ink(
        decoration: BoxDecoration(
          gradient: const LinearGradient(
              colors: [Color(0xFF00B4D8), Color(0xFF0066FF)]),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
                color: kAccent.o(0.4),
                blurRadius: 20,
                offset: const Offset(0, 4))
          ],
        ),
        child: Container(
          alignment: Alignment.center,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.biotech_rounded, color: Colors.white, size: 22),
              const SizedBox(width: 10),
              Text('Phân tích X-quang',
                  style: GoogleFonts.inter(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: Colors.white)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _loadingButton() {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [
          kAccent.o(0.5),
          const Color(0xFF0066FF).o(0.5)
        ]),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor:
                    AlwaysStoppedAnimation<Color>(Colors.white.o(0.8))),
          ),
          const SizedBox(width: 12),
          Text('Đang phân tích...',
              style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70)),
        ],
      ),
    );
  }

  Widget _buildFormatHint() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0A2A3A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: kAccent.o(0.12)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(Icons.info_outline_rounded, color: kAccent, size: 16),
            const SizedBox(width: 8),
            Text('Hướng dẫn',
                style: GoogleFonts.inter(
                    color: kAccent,
                    fontSize: 13,
                    fontWeight: FontWeight.w600)),
          ]),
          const SizedBox(height: 10),
          _hintItem('Sử dụng ảnh X-quang ngực chính diện (PA view)'),
          _hintItem('Ảnh nên có độ phân giải tối thiểu 224×224px'),
          _hintItem('Định dạng JPG, PNG hoặc WEBP được hỗ trợ'),
          _hintItem(
              'Kết quả chỉ mang tính tham khảo, không thay thế bác sĩ'),
        ],
      ),
    );
  }

  Widget _hintItem(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 5),
            child: Container(
                width: 5,
                height: 5,
                decoration: const BoxDecoration(
                    shape: BoxShape.circle, color: kAccent)),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: GoogleFonts.inter(
                    fontSize: 12, color: Colors.white54, height: 1.5)),
          ),
        ],
      ),
    );
  }

  // ── Result Panel ─────────────────────────────────────────────
  Widget _buildResultPanel() {
    return _glassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(Icons.analytics_rounded, 'Kết quả phân tích'),
          const SizedBox(height: 20),
          if (_result == null && _errorMessage == null && !_isAnalyzing)
            _buildEmptyResult(),
          if (_isAnalyzing) _buildLoadingResult(),
          if (_errorMessage != null) _buildErrorResult(),
          if (_result != null) _buildResultContent(),
        ],
      ),
    ).animate().fadeIn(delay: 400.ms, duration: 600.ms).slideX(begin: 0.1, end: 0);
  }

  Widget _buildEmptyResult() {
    return Container(
      height: 320,
      alignment: Alignment.center,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.biotech_outlined,
              size: 64, color: Colors.white.o(0.08)),
          const SizedBox(height: 16),
          Text('Chưa có kết quả',
              style: GoogleFonts.inter(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.white24)),
          const SizedBox(height: 8),
          Text('Tải lên ảnh X-quang và\nnhấn "Phân tích" để bắt đầu',
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                  fontSize: 13, color: Colors.white24, height: 1.5)),
        ],
      ),
    );
  }

  Widget _buildLoadingResult() {
    return Container(
      height: 320,
      alignment: Alignment.center,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Stack(
            alignment: Alignment.center,
            children: [
              SizedBox(
                width: 80,
                height: 80,
                child: CircularProgressIndicator(
                    strokeWidth: 3,
                    valueColor: AlwaysStoppedAnimation<Color>(kAccent.o(0.3))),
              ),
              const SizedBox(
                width: 64,
                height: 64,
                child: CircularProgressIndicator(
                    strokeWidth: 3,
                    valueColor:
                        AlwaysStoppedAnimation<Color>(kAccent)),
              ),
              const Icon(Icons.psychology_rounded, color: kAccent, size: 28),
            ],
          ).animate(onPlay: (c) => c.repeat()).rotate(
              duration: 3.seconds, curve: Curves.linear),
          const SizedBox(height: 24),
          Text('AI đang phân tích ảnh...',
              style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70)),
          const SizedBox(height: 8),
          Text('Mô hình EfficientNet-B5 đang xử lý',
              style: GoogleFonts.inter(fontSize: 12, color: Colors.white38)),
        ],
      ),
    );
  }

  Widget _buildErrorResult() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kRed.o(0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: kRed.o(0.25)),
      ),
      child: Column(
        children: [
          Icon(Icons.error_outline_rounded, color: kRed, size: 40),
          const SizedBox(height: 12),
          Text('Lỗi kết nối',
              style: GoogleFonts.inter(
                  color: kRed, fontWeight: FontWeight.w700, fontSize: 16)),
          const SizedBox(height: 8),
          Text(_errorMessage!,
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                  color: Colors.white60, fontSize: 13, height: 1.5)),
          const SizedBox(height: 16),
          OutlinedButton(
            onPressed: () => setState(() => _errorMessage = null),
            style: OutlinedButton.styleFrom(
              foregroundColor: kRed,
              side: BorderSide(color: kRed.o(0.4)),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10)),
            ),
            child: const Text('Thử lại'),
          ),
        ],
      ),
    );
  }

  Widget _buildResultContent() {
    final res = _result!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildDiagnosisCard(res),
        const SizedBox(height: 20),
        Text('Chi tiết từng loại bệnh',
            style: GoogleFonts.inter(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.white70)),
        const SizedBox(height: 12),
        ...res.results.asMap().entries.map((entry) {
          return _buildDiseaseBar(entry.value, entry.key)
              .animate()
              .fadeIn(delay: (entry.key * 80).ms, duration: 400.ms)
              .slideX(begin: 0.15, end: 0);
        }),
        const SizedBox(height: 16),
        _buildDisclaimer(),
      ],
    );
  }

  Widget _buildDiagnosisCard(PredictionResponse res) {
    final severityColor = _severityColor(res.predictedSeverity);
    final severityLabel = _severityLabel(res.predictedSeverity);
    final severityIcon = _severityIcon(res.predictedSeverity);
    final top = res.results.first;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [severityColor.o(0.12), severityColor.o(0.04)],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: severityColor.o(0.35), width: 1.5),
        boxShadow: [
          BoxShadow(color: severityColor.o(0.1), blurRadius: 24)
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: severityColor.o(0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(severityIcon, color: severityColor, size: 24),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Kết luận chẩn đoán',
                        style: GoogleFonts.inter(
                            fontSize: 11,
                            color: Colors.white38,
                            fontWeight: FontWeight.w500,
                            letterSpacing: 0.5)),
                    const SizedBox(height: 2),
                    Text(res.predictedNameVi,
                        style: GoogleFonts.inter(
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                            letterSpacing: -0.5)),
                  ],
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: severityColor.o(0.15),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: severityColor.o(0.4)),
                ),
                child: Text(severityLabel,
                    style: GoogleFonts.inter(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: severityColor)),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _confidenceBadge('${res.confidence.toStringAsFixed(1)}%',
                  'Độ tin cậy', severityColor),
              const SizedBox(width: 12),
              _confidenceBadge(top.description, 'Mô tả', kAccent),
            ],
          ),
        ],
      ),
    ).animate().fadeIn(duration: 500.ms).scale(begin: const Offset(0.97, 0.97));
  }

  Widget _confidenceBadge(String value, String label, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
        decoration: BoxDecoration(
          color: color.o(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.o(0.2)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label,
                style: GoogleFonts.inter(
                    fontSize: 10,
                    color: Colors.white38,
                    letterSpacing: 0.3)),
            const SizedBox(height: 3),
            Text(value,
                style: GoogleFonts.inter(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: Colors.white),
                overflow: TextOverflow.ellipsis),
          ],
        ),
      ),
    );
  }

  Widget _buildDiseaseBar(DiseaseResult disease, int index) {
    final isTop = index == 0;
    final barColor = disease.colorValue;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isTop ? barColor.o(0.08) : Colors.white.o(0.03),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color: isTop ? barColor.o(0.25) : Colors.white.o(0.06)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              if (isTop)
                Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding: const EdgeInsets.all(3),
                  decoration: BoxDecoration(
                      color: barColor.o(0.2),
                      borderRadius: BorderRadius.circular(6)),
                  child: Icon(Icons.star_rounded, size: 14, color: barColor),
                ),
              Expanded(
                child: Text(
                  disease.nameVi,
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    fontWeight:
                        isTop ? FontWeight.w700 : FontWeight.w500,
                    color: isTop ? Colors.white : Colors.white70,
                  ),
                ),
              ),
              Text(
                '${disease.percentage.toStringAsFixed(2)}%',
                style: GoogleFonts.inter(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  color: isTop ? barColor : Colors.white60,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: disease.probability,
              minHeight: 8,
              backgroundColor: Colors.white.o(0.06),
              valueColor: AlwaysStoppedAnimation<Color>(barColor),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: kOrange.o(0.06),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: kOrange.o(0.2)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded, color: kOrange, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Kết quả này chỉ mang tính tham khảo hỗ trợ. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để được chẩn đoán chính xác.',
              style: GoogleFonts.inter(
                  fontSize: 12, color: kOrange.o(0.85), height: 1.5),
            ),
          ),
        ],
      ),
    );
  }

  // ── Footer ───────────────────────────────────────────────────
  Widget _buildFooter() {
    return Container(
      margin: const EdgeInsets.only(top: 60),
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: kBgCard.o(0.5),
        border: Border(
            top: BorderSide(color: kAccent.o(0.1), width: 1)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.medical_services_rounded, color: kAccent, size: 20),
              const SizedBox(width: 8),
              Text('MediScan AI',
                  style: GoogleFonts.inter(
                      color: Colors.white70,
                      fontWeight: FontWeight.w600,
                      fontSize: 16)),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Ứng dụng phân tích ảnh X-quang ngực bằng trí tuệ nhân tạo\nModel: EfficientNet-B5 · Dataset: VinBigData + COVID-19 Radiography',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
                color: Colors.white30, fontSize: 12, height: 1.6),
          ),
        ],
      ),
    );
  }

  // ── Helpers ──────────────────────────────────────────────────
  Widget _glassCard({required Widget child}) => Container(
        padding: const EdgeInsets.all(28),
        decoration: BoxDecoration(
          color: kBgCard,
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: Colors.white.o(0.07)),
          boxShadow: [
            BoxShadow(
                color: Colors.black.o(0.3),
                blurRadius: 30,
                offset: const Offset(0, 8))
          ],
        ),
        child: child,
      );

  Widget _sectionTitle(IconData icon, String title) => Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: kAccent.o(0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: kAccent, size: 20),
          ),
          const SizedBox(width: 12),
          Text(title,
              style: GoogleFonts.inter(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                  letterSpacing: -0.3)),
        ],
      );

  Color _severityColor(String severity) {
    switch (severity) {
      case 'danger':
        return kRed;
      case 'warning':
        return kOrange;
      case 'normal':
        return kGreen;
      default:
        return kAccent;
    }
  }

  String _severityLabel(String severity) {
    switch (severity) {
      case 'danger':
        return 'NGUY HIỂM';
      case 'warning':
        return 'CẦN THEO DÕI';
      case 'normal':
        return 'BÌNH THƯỜNG';
      default:
        return 'KHÔNG XÁC ĐỊNH';
    }
  }

  IconData _severityIcon(String severity) {
    switch (severity) {
      case 'danger':
        return Icons.dangerous_rounded;
      case 'warning':
        return Icons.medical_information_rounded;
      case 'normal':
        return Icons.check_circle_rounded;
      default:
        return Icons.help_rounded;
    }
  }
}
