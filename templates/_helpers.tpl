{{/* Common name + labels for the shared semantic memory chart. */}}

{{- define "memory-mcp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "memory-mcp.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- include "memory-mcp.name" . | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "memory-mcp.postgres.fullname" -}}
{{- printf "%s-postgres" (include "memory-mcp.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "memory-mcp.labels" -}}
app.kubernetes.io/name: {{ include "memory-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: selamy-agents
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "memory-mcp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "memory-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "memory-mcp.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}

{{/* The Postgres DSN the server reads (password from the synced Secret). */}}
{{- define "memory-mcp.pgHost" -}}
{{ include "memory-mcp.postgres.fullname" . }}.{{ .Release.Namespace }}.svc
{{- end -}}
