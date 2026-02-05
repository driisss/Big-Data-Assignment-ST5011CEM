"""
Flexible XML to CSV Converter for Bus Data
Extracts ALL available fields regardless of structure
Prioritizes travel time prediction data but captures everything
"""

import xml.etree.ElementTree as ET
import csv
import os
from datetime import datetime
from collections import defaultdict
import re

class XMLtoCSVConverter:
    def __init__(self):
        self.all_fields = set()
        self.records = []
        self.field_stats = defaultdict(int)
        self.namespace = {}
        
    def extract_namespace(self, root):
        """Extract XML namespace for proper tag handling"""
        match = re.match(r'\{.*\}', root.tag)
        if match:
            ns = match.group(0)
            self.namespace = {'ns': ns[1:-1]}
            return ns
        return ''
    
    def parse_stop_points(self, root, ns):
        """Extract stop point metadata"""
        stop_data = {}
        
        # Find StopPoints section
        stop_points = root.findall('.//ns:StopPoints/ns:AnnotatedStopPointRef', self.namespace) if ns else \
                     root.findall('.//StopPoints/AnnotatedStopPointRef')
        
        for stop in stop_points:
            stop_id = self._get_text(stop, 'StopPointRef', ns)
            if stop_id:
                stop_data[stop_id] = {
                    'StopPointRef': stop_id,
                    'CommonName': self._get_text(stop, 'CommonName', ns),
                    'Indicator': self._get_text(stop, 'Indicator', ns),
                    'LocalityName': self._get_text(stop, 'LocalityName', ns),
                    'LocalityQualifier': self._get_text(stop, 'LocalityQualifier', ns),
                    'Latitude': self._get_text(stop, 'Latitude', ns),
                    'Longitude': self._get_text(stop, 'Longitude', ns),
                }
        
        return stop_data
    
    def parse_route_links(self, root, ns, stop_data):
        """Extract route link data (stop-to-stop connections with distance)"""
        records = []
        
        route_links = root.findall('.//ns:RouteLink', self.namespace) if ns else \
                     root.findall('.//RouteLink')
        
        for link in route_links:
            link_id = link.get('id', '')
            from_stop = self._get_text(link.find('ns:From/ns:StopPointRef' if ns else 'From/StopPointRef', self.namespace), '', ns)
            to_stop = self._get_text(link.find('ns:To/ns:StopPointRef' if ns else 'To/StopPointRef', self.namespace), '', ns)
            distance = self._get_text(link, 'Distance', ns)
            
            # Get route section ID from parent
            route_section = link.find('..')
            route_section_id = route_section.get('id', '') if route_section is not None else ''
            
            record = {
                'RecordType': 'RouteLink',
                'RouteLinkID': link_id,
                'RouteSectionID': route_section_id,
                'FromStopID': from_stop,
                'ToStopID': to_stop,
                'Distance_m': distance,
            }
            
            # Enrich with stop details
            if from_stop in stop_data:
                for key, val in stop_data[from_stop].items():
                    if key != 'StopPointRef':
                        record[f'From_{key}'] = val
            
            if to_stop in stop_data:
                for key, val in stop_data[to_stop].items():
                    if key != 'StopPointRef':
                        record[f'To_{key}'] = val
            
            records.append(record)
            self._update_field_stats(record)
        
        return records
    
    def parse_journey_patterns(self, root, ns, stop_data):
        """Extract journey pattern timing links (with runtime between stops)"""
        records = []
        
        timing_links = root.findall('.//ns:JourneyPatternTimingLink', self.namespace) if ns else \
                      root.findall('.//JourneyPatternTimingLink')
        
        for link in timing_links:
            link_id = link.get('id', '')
            
            # From stop details
            from_elem = link.find('ns:From' if ns else 'From', self.namespace)
            from_stop = self._get_text(from_elem, 'StopPointRef', ns)
            from_seq = from_elem.get('SequenceNumber', '') if from_elem is not None else ''
            from_activity = self._get_text(from_elem, 'Activity', ns)
            from_timing = self._get_text(from_elem, 'TimingStatus', ns)
            
            # To stop details
            to_elem = link.find('ns:To' if ns else 'To', self.namespace)
            to_stop = self._get_text(to_elem, 'StopPointRef', ns)
            to_seq = to_elem.get('SequenceNumber', '') if to_elem is not None else ''
            to_activity = self._get_text(to_elem, 'Activity', ns)
            to_timing = self._get_text(to_elem, 'TimingStatus', ns)
            
            # Runtime and other timing info
            runtime = self._get_text(link, 'RunTime', ns)
            wait_time = self._get_text(link, 'WaitTime', ns)
            route_link_ref = self._get_text(link, 'RouteLinkRef', ns)
            
            # Get parent journey pattern section
            parent = link.find('..')
            jps_id = parent.get('id', '') if parent is not None else ''
            
            record = {
                'RecordType': 'JourneyPatternTimingLink',
                'TimingLinkID': link_id,
                'JourneyPatternSectionID': jps_id,
                'FromStopID': from_stop,
                'FromSequenceNumber': from_seq,
                'FromActivity': from_activity,
                'FromTimingStatus': from_timing,
                'ToStopID': to_stop,
                'ToSequenceNumber': to_seq,
                'ToActivity': to_activity,
                'ToTimingStatus': to_timing,
                'RunTime': runtime,
                'RunTime_Seconds': self._parse_duration(runtime),
                'WaitTime': wait_time,
                'RouteLinkRef': route_link_ref,
            }
            
            # Enrich with stop details
            if from_stop in stop_data:
                for key, val in stop_data[from_stop].items():
                    if key != 'StopPointRef':
                        record[f'From_{key}'] = val
            
            if to_stop in stop_data:
                for key, val in stop_data[to_stop].items():
                    if key != 'StopPointRef':
                        record[f'To_{key}'] = val
            
            records.append(record)
            self._update_field_stats(record)
        
        return records
    
    def parse_vehicle_journeys(self, root, ns):
        """Extract vehicle journey schedules"""
        records = []
        
        journeys = root.findall('.//ns:VehicleJourney', self.namespace) if ns else \
                  root.findall('.//VehicleJourney')
        
        for journey in journeys:
            # Basic journey info
            journey_code = self._get_text(journey, 'PrivateCode', ns)
            departure_time = self._get_text(journey, 'DepartureTime', ns)
            journey_pattern_ref = self._get_text(journey, 'JourneyPatternRef', ns)
            line_ref = self._get_text(journey, 'LineRef', ns)
            service_ref = self._get_text(journey, 'ServiceRef', ns)
            operator_ref = self._get_text(journey, 'OperatorRef', ns)
            
            # Operating profile
            operating_profile = journey.find('ns:OperatingProfile' if ns else 'OperatingProfile', self.namespace)
            days_of_week = []
            if operating_profile is not None:
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    if operating_profile.find(f'ns:RegularDayType/ns:{day}' if ns else f'RegularDayType/{day}', self.namespace) is not None:
                        days_of_week.append(day)
            
            record = {
                'RecordType': 'VehicleJourney',
                'JourneyCode': journey_code,
                'DepartureTime': departure_time,
                'JourneyPatternRef': journey_pattern_ref,
                'LineRef': line_ref,
                'ServiceRef': service_ref,
                'OperatorRef': operator_ref,
                'DaysOfWeek': ','.join(days_of_week) if days_of_week else '',
            }
            
            records.append(record)
            self._update_field_stats(record)
        
        return records
    
    def parse_services(self, root, ns):
        """Extract service metadata"""
        records = []
        
        services = root.findall('.//ns:Service', self.namespace) if ns else \
                  root.findall('.//Service')
        
        for service in services:
            service_code = self._get_text(service, 'ServiceCode', ns)
            
            # Lines
            lines = service.findall('ns:Lines/ns:Line' if ns else 'Lines/Line', self.namespace)
            for line in lines:
                line_id = line.get('id', '')
                line_name = self._get_text(line, 'LineName', ns)
                
                record = {
                    'RecordType': 'Service',
                    'ServiceCode': service_code,
                    'LineID': line_id,
                    'LineName': line_name,
                }
                
                # Operating period
                op_period = service.find('ns:OperatingPeriod' if ns else 'OperatingPeriod', self.namespace)
                if op_period is not None:
                    record['StartDate'] = self._get_text(op_period, 'StartDate', ns)
                    record['EndDate'] = self._get_text(op_period, 'EndDate', ns)
                
                # Operator
                record['Operator'] = self._get_text(service, 'RegisteredOperatorRef', ns)
                
                records.append(record)
                self._update_field_stats(record)
        
        return records
    
    def parse_operators(self, root, ns):
        """Extract operator information"""
        records = []
        
        operators = root.findall('.//ns:Operator', self.namespace) if ns else \
                   root.findall('.//Operator')
        
        for op in operators:
            record = {
                'RecordType': 'Operator',
                'OperatorID': op.get('id', ''),
                'OperatorCode': self._get_text(op, 'OperatorCode', ns),
                'OperatorName': self._get_text(op, 'OperatorNameOnLicence', ns) or self._get_text(op, 'OperatorShortName', ns),
                'TradingName': self._get_text(op, 'TradingName', ns),
            }
            
            records.append(record)
            self._update_field_stats(record)
        
        return records
    
    def _get_text(self, element, tag, ns):
        """Safely get text from XML element"""
        if element is None:
            return ''
        
        if tag:
            child = element.find(f'ns:{tag}' if ns else tag, self.namespace)
            return child.text.strip() if child is not None and child.text else ''
        else:
            return element.text.strip() if element.text else ''
    
    def _parse_duration(self, duration_str):
        """Convert ISO 8601 duration to seconds"""
        if not duration_str:
            return ''
        
        # Parse PT format (e.g., PT1M20S)
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return str(hours * 3600 + minutes * 60 + seconds)
        
        return ''
    
    def _update_field_stats(self, record):
        """Track which fields have data"""
        for key, val in record.items():
            if val and str(val).strip():
                self.field_stats[key] += 1
    
    def process_file(self, xml_file, output_csv=None):
        """Process a single XML file and extract all data"""
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(xml_file)}")
        print(f"{'='*80}")
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract namespace
            ns = self.extract_namespace(root)
            
            # Parse all sections
            print("\n[1/7] Extracting stop points...")
            stop_data = self.parse_stop_points(root, ns)
            print(f"   Found {len(stop_data)} stop points")
            
            print("\n[2/7] Extracting route links (stop-to-stop connections)...")
            route_links = self.parse_route_links(root, ns, stop_data)
            print(f"   Found {len(route_links)} route links")
            
            print("\n[3/7] Extracting journey pattern timing links (with runtime)...")
            timing_links = self.parse_journey_patterns(root, ns, stop_data)
            print(f"   Found {len(timing_links)} timing links")
            
            print("\n[4/7] Extracting vehicle journeys...")
            journeys = self.parse_vehicle_journeys(root, ns)
            print(f"   Found {len(journeys)} vehicle journeys")
            
            print("\n[5/7] Extracting services...")
            services = self.parse_services(root, ns)
            print(f"   Found {len(services)} service records")
            
            print("\n[6/7] Extracting operators...")
            operators = self.parse_operators(root, ns)
            print(f"   Found {len(operators)} operators")
            
            # Combine all records
            print("\n[7/7] Combining all records...")
            all_records = route_links + timing_links + journeys + services + operators
            
            # Add source filename to each record
            filename = os.path.basename(xml_file)
            for record in all_records:
                record['SourceFile'] = filename
                self.all_fields.update(record.keys())
            
            self.records.extend(all_records)
            
            print(f"\n✓ Total records extracted: {len(all_records)}")
            print(f"✓ Total unique fields: {len(self.all_fields)}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error processing {xml_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, directory):
        """Process all XML files in a directory"""
        xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]
        
        print(f"\nFound {len(xml_files)} XML files to process")
        print(f"Directory: {directory}")
        
        success_count = 0
        for xml_file in xml_files:
            full_path = os.path.join(directory, xml_file)
            if self.process_file(full_path):
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"Processing Summary: {success_count}/{len(xml_files)} files processed successfully")
        print(f"{'='*80}")
    
    def write_csv(self, output_csv):
        """Write all records to CSV"""
        if not self.records:
            print("\n✗ No records to write!")
            return
        
        # Sort fields by priority
        priority_fields = [
            'SourceFile', 'RecordType',
            'FromStopID', 'ToStopID', 
            'From_CommonName', 'To_CommonName',
            'FromSequenceNumber', 'ToSequenceNumber',
            'RunTime', 'RunTime_Seconds',
            'Distance_m',
            'From_Latitude', 'From_Longitude',
            'To_Latitude', 'To_Longitude',
        ]
        
        # Get all fields, prioritizing important ones
        sorted_fields = []
        for field in priority_fields:
            if field in self.all_fields:
                sorted_fields.append(field)
        
        # Add remaining fields
        for field in sorted(self.all_fields):
            if field not in sorted_fields:
                sorted_fields.append(field)
        
        print(f"\n{'='*80}")
        print(f"Writing CSV: {output_csv}")
        print(f"{'='*80}")
        print(f"Total records: {len(self.records)}")
        print(f"Total fields: {len(sorted_fields)}")
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fields)
            writer.writeheader()
            writer.writerows(self.records)
        
        print(f"\n✓ CSV file created successfully!")
        
        # Print data quality report
        self._print_data_quality_report(sorted_fields)
    
    def _print_data_quality_report(self, fields):
        """Print comprehensive data quality report"""
        total_records = len(self.records)
        
        print(f"\n{'='*80}")
        print("DATA QUALITY REPORT")
        print(f"{'='*80}")
        
        # Count records by type
        type_counts = defaultdict(int)
        for record in self.records:
            type_counts[record.get('RecordType', 'Unknown')] += 1
        
        print("\nRecord Type Distribution:")
        for rec_type, count in sorted(type_counts.items()):
            pct = (count / total_records * 100) if total_records > 0 else 0
            print(f"  {rec_type:30s}: {count:6d} ({pct:5.1f}%)")
        
        print("\n" + "="*80)
        print("FIELD COMPLETENESS (High Priority Fields)")
        print("="*80)
        
        # Priority field completeness
        priority_fields_check = {
            'Stop-to-Stop Data': ['FromStopID', 'ToStopID', 'From_CommonName', 'To_CommonName'],
            'Runtime Data': ['RunTime', 'RunTime_Seconds'],
            'Coordinates': ['From_Latitude', 'From_Longitude', 'To_Latitude', 'To_Longitude'],
            'Distance': ['Distance_m'],
            'Sequence': ['FromSequenceNumber', 'ToSequenceNumber'],
            'Timing Info': ['DepartureTime', 'TimingStatus'],
        }
        
        for category, field_list in priority_fields_check.items():
            print(f"\n{category}:")
            for field in field_list:
                if field in self.field_stats:
                    count = self.field_stats[field]
                    pct = (count / total_records * 100) if total_records > 0 else 0
                    status = "✓" if pct > 50 else "⚠" if pct > 10 else "✗"
                    print(f"  {status} {field:30s}: {count:6d} records ({pct:5.1f}%)")
                else:
                    print(f"  ✗ {field:30s}: NOT FOUND")
        
        # Overall completeness
        print(f"\n{'='*80}")
        print("OVERALL DATA ASSESSMENT")
        print(f"{'='*80}")
        
        # Check what we have
        has_runtime = self.field_stats.get('RunTime_Seconds', 0) > 0
        has_stops = self.field_stats.get('FromStopID', 0) > 0 and self.field_stats.get('ToStopID', 0) > 0
        has_coords = self.field_stats.get('From_Latitude', 0) > 0 or self.field_stats.get('To_Latitude', 0) > 0
        has_distance = self.field_stats.get('Distance_m', 0) > 0
        
        print("\nData Availability:")
        print(f"  {'✓' if has_stops else '✗'} Stop-to-stop connections")
        print(f"  {'✓' if has_runtime else '✗'} Runtime between stops (ideal for ML)")
        print(f"  {'✓' if has_distance else '✗'} Distance between stops")
        print(f"  {'✓' if has_coords else '✗'} GPS coordinates")
        
        print("\nML Readiness Assessment:")
        if has_runtime and has_stops:
            print("  ✓✓✓ EXCELLENT - Direct travel time prediction possible")
            print("      Has runtime, stops, and sequences")
        elif has_distance and has_stops:
            print("  ✓✓ GOOD - Can estimate travel times from distance")
            print("      Has stop-to-stop distance data")
        elif has_stops:
            print("  ✓ FAIR - Basic route data available")
            print("      Can construct networks but missing time/distance")
        else:
            print("  ⚠ LIMITED - Only schedule/metadata available")
        
        print(f"\n{'='*80}")
        print("Top 20 Most Complete Fields:")
        print(f"{'='*80}")
        sorted_stats = sorted(self.field_stats.items(), key=lambda x: x[1], reverse=True)[:20]
        for field, count in sorted_stats:
            pct = (count / total_records * 100) if total_records > 0 else 0
            print(f"  {field:35s}: {count:6d} ({pct:5.1f}%)")
        
        print(f"\n{'='*80}\n")


def main():
    """Main execution function"""
    import sys
    
    print("="*80)
    print("FLEXIBLE XML TO CSV CONVERTER FOR BUS DATA")
    print("Extracts ALL available fields regardless of structure")
    print("="*80)
    
    # Get input path
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Default to current directory
        input_path = os.path.dirname(os.path.abspath(__file__))
    
    # Check if input is file or directory
    if os.path.isfile(input_path) and input_path.endswith('.xml'):
        # Single file
        xml_dir = os.path.dirname(input_path)
        output_csv = os.path.join(xml_dir, 'bus_data_extracted.csv')
        
        converter = XMLtoCSVConverter()
        converter.process_file(input_path, output_csv)
        converter.write_csv(output_csv)
        
    elif os.path.isdir(input_path):
        # Directory of XML files
        # Check for XML subdirectory
        xml_subdir = os.path.join(input_path, 'Metroline_TfL_9w6A35W')
        if os.path.isdir(xml_subdir):
            xml_dir = xml_subdir
        else:
            xml_dir = input_path
        
        output_csv = os.path.join(input_path, 'bus_data_all_files.csv')
        
        converter = XMLtoCSVConverter()
        converter.process_directory(xml_dir)
        converter.write_csv(output_csv)
        
    else:
        print(f"Error: Invalid input path: {input_path}")
        print("\nUsage:")
        print("  python xml_to_csv_flexible.py [path_to_xml_file_or_directory]")
        print("\nIf no path provided, processes current directory")
        sys.exit(1)
    
    print("\n✓ All done!")
    print(f"✓ CSV saved to: {output_csv}")


if __name__ == '__main__':
    main()
