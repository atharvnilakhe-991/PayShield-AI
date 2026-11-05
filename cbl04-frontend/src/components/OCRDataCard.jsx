import {
  FileText,
  Hash,
  IndianRupee,
  Calendar,
  User,
  Store,
  Sparkles,
  Info,
  CheckCircle
} from 'lucide-react';

/**
 * OCRDataCard displays the parsed payment metadata from response.ocr_data.
 * Handles null, undefined, or empty values with truthful 'Not detected' indicators.
 */
export default function OCRDataCard({ ocrData, isLive = true }) {
  const data = ocrData || {};

  const resolvedTxnId = data.transaction_id || data.google_txn_id || null;
  const resolvedDate = data.date
    ? (data.time ? `${data.date} at ${data.time}` : data.date)
    : (data.time || null);
  const resolvedSender = data.sender
    ? (data.sender_upi ? `${data.sender} (${data.sender_upi})` : data.sender)
    : (data.sender_upi || null);
  const resolvedReceiver = data.receiver
    ? (data.receiver_upi ? `${data.receiver} (${data.receiver_upi})` : data.receiver)
    : (data.receiver_upi || null);

  // Ensure the first symbol is always Rupee (₹) when displaying transaction amount
  let formattedAmount = null;
  if (data.amount && typeof data.amount === 'string') {
    const trimmed = data.amount.trim();
    if (trimmed.startsWith('₹')) {
      formattedAmount = trimmed;
    } else {
      const cleanNum = trimmed.replace(/^(?:[₹\$\?\{\=\[\]]|Rs\.?|INR)\s*/i, '');
      formattedAmount = `₹${cleanNum}`;
    }
  } else if (data.amount_numeric !== null && data.amount_numeric !== undefined) {
    formattedAmount = `₹${data.amount_numeric}`;
  }

  const fields = [
    {
      key: 'transaction_id',
      label: 'Transaction ID / UPI Ref',
      value: resolvedTxnId,
      icon: Hash,
      monospace: true,
    },
    {
      key: 'amount',
      label: 'Transaction Amount',
      value: formattedAmount,
      icon: IndianRupee,
      highlight: true,
    },
    {
      key: 'status',
      label: 'Transaction Status',
      value: data.status,
      icon: CheckCircle,
    },
    {
      key: 'date',
      label: 'Timestamp / Date',
      value: resolvedDate,
      icon: Calendar,
    },
    {
      key: 'sender',
      label: 'Sender Name / UPI ID',
      value: resolvedSender,
      icon: User,
    },
    {
      key: 'receiver',
      label: 'Receiver / Merchant',
      value: resolvedReceiver,
      icon: Store,
    },
  ];

  return (
    <div className="card ocr-card">
      <div className="card-header ocr-card-header">
        <div className="card-header-left">
          <FileText className="card-icon text-primary" size={20} />
          <h3>{isLive ? 'Extracted OCR Metadata' : 'Simulated OCR Data'}</h3>
        </div>
        <span className="badge badge-accent">
          <Sparkles size={12} />
          {isLive ? 'EasyOCR Engine (Live)' : 'Offline OCR'}
        </span>
      </div>

      <div className="card-body">
        <p className="ocr-intro-text">
          {isLive
            ? 'Text extracted from receipt screenshot layout using PyTorch EasyOCR deep neural models.'
            : 'Text extracted from receipt layout using simulated OCR pipeline.'}
        </p>

        <div className="ocr-grid">
          {fields.map(({ key, label, value, icon: Icon, monospace, highlight }) => {
            const hasValue = value !== null && value !== undefined && String(value).trim() !== '';
            return (
              <div key={key} className={`ocr-field-item ${highlight ? 'ocr-field-highlight' : ''}`}>
                <div className="ocr-field-label-group">
                  <Icon size={14} className="ocr-field-icon" />
                  <span className="ocr-field-label">{label}</span>
                </div>
                <div className={`ocr-field-value ${monospace ? 'ocr-monospace' : ''}`}>
                  {hasValue ? (
                    <span>{value}</span>
                  ) : (
                    <span className="ocr-value-missing" title="Field not detected in receipt screenshot">
                      Not detected
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="ocr-footer-note">
          <Info size={13} className="text-secondary" />
          <span>
            {isLive
              ? 'Visible text extracted for semantic audit; does not confirm bank-side ledger settlement.'
              : 'Synthetic demo data for offline testing.'}
          </span>
        </div>
      </div>
    </div>
  );
}
