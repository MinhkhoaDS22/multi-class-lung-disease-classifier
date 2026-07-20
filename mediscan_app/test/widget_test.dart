import 'package:flutter_test/flutter_test.dart';
import 'package:mediscan_app/main.dart';

void main() {
  testWidgets('MediScan AI smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const MediScanApp());
    expect(find.text('MediScan AI'), findsWidgets);
  });
}
